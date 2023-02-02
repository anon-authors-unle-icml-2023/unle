import logging
import math
import pickle
from time import time
from typing import (Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple,
                    Union, cast)

import numpy as np
import numpyro.distributions as npdist
import pyro.distributions as pdist
import torch
import torch.distributions as tdist
from abcpy.backends import BackendDummy
from abcpy.NN_utilities.utilities import save_net
from abcpy.statistics import Identity
from abcpy.statisticslearning import \
    ExponentialFamilyScoreMatching as ExpFamStatistics
from abcpy.statisticslearning import StatisticsLearning
from abcpy.transformers import BoundedVarScaler, MinMaxScaler
from optax._src.transform import trace
from sbi import inference as inference
from sbibm.algorithms.sbi.utils import wrap_prior_dist, wrap_simulator_fn
from sbibm.tasks.task import Task
from smnle.src.exchange_mcmc import exchange_MCMC_with_SM_statistics
from smnle.src.networks import createDefaultNN, createDefaultNNWithDerivatives
from torch import nn
from sbi_ebm.distributions import DoublyIntractableLogDensity, maybe_wrap

from sbi_ebm.pytypes import Array, DoublyIntractableLogDensity_T, PRNGKeyArray
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithmFactory, MCMCConfig
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.samplers.kernels.rwmh import RWConfig, RWKernel, RWKernelFactory
from sbi_ebm.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from sbi_ebm.sbi_ebm import TaskConfig
from sbi_ebm.sbibm.jax_torch_interop import JaxExpFamLikelihood
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist
from pyro.distributions import transforms as pyro_transforms

from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from .jax_torch_interop import _JaxExpFamLikelihoodDist

from jax import jit, random
import jax.numpy as jnp
from sbi_ebm.sbibm.sbi_ebm import _evaluate_posterior
from flax import struct

def scale_val(scaler, vals, requires_grad=False):
    if scaler is None:
        return vals
    return torch.tensor(
        scaler.transform(vals).astype("float32"), requires_grad=requires_grad
    )


class SMNLESingleRoundTrainResults(NamedTuple):
    statistics: StatisticsLearning
    net_data: nn.Module
    net_theta: nn.Module
    scaler_data: np_transforms.Transform
    scaler_theta: np_transforms.Transform
    posterior_samples: Array
    train_theta: Array
    train_data: Array
    jax_log_likelihood: Callable[[Array, Array], Array]

    def log_likelihood(self, theta, x):
        theta, x = torch.atleast_2d(torch.zeros((2,))), torch.atleast_2d(x)
        identity = lambda x: x
        scaler_theta = lambda x: scale_val(self.scaler_theta, x)
        scaler_data = lambda x: scale_val(self.scaler_data, x)

        theta_post_net = self.net_theta(scaler_theta(theta))

        nat_param = torch.cat((theta_post_net, torch.ones_like(theta[..., :1])), axis=-1)  # type: ignore
        suff_stat = self.net_data(scaler_data(x))

        assert nat_param.shape[0] == 1
        assert suff_stat.shape[0] == 1

        return torch.dot(suff_stat[0], nat_param[0])


class SMNLEConfig(struct.PyTreeNode):
    task: TaskConfig
    num_observation: Optional[int] = None
    num_rounds: int = 1
    observation: Optional[torch.Tensor] = None
    num_simulations: int = 1000
    automatic_transforms_enabled: bool = True
    num_samples: int = 1000
    # SM training arg
    technique: Literal["SM", "SSM"] = "SM"
    epochs: int = 500
    no_scheduler: bool = False
    noise_sliced: str = "radermacher"
    no_var_red_sliced: bool = False
    no_bn: bool = False
    affine_batch_norm: bool = False
    lr_data: float = 0.001
    SM_lr_theta: float = 0.001
    batch_size: int = 1000
    no_early_stop: bool = False
    update_batchnorm_running_means_before_eval: bool = False
    momentum: float = 0.9
    epochs_before_early_stopping: int = 200
    epochs_test_interval: int = 10
    scale_samples: bool = True
    scale_parameters: bool = True
    seed: int = 42
    cuda: bool = False
    lam: int = 0
    num_chains: int = 100
    thinning_factor: int =10
    propose_new_theta_exchange_MCMC: Literal["transformation", "adaptive_transformation", "norm"] = "transformation"
    burnin_exchange_MCMC: int = 300
    tuning_window_exchange_MCMC: int = 100
    aux_MCMC_inner_steps_exchange_MCMC: int = 100
    aux_MCMC_proposal_size_exchange_MCMC: float = 0.1
    proposal_size_exchange_MCMC: float = 0.1
    bridging_exch_MCMC: int = 0
    debug_level: int = 100
    theta_vect: Optional[torch.Tensor] = None
    use_jax_mcmc: bool = False
    use_data_from_past_rounds: bool = True
    evaluate_posterior: bool = False


class SMNLETrainResults(NamedTuple):
    single_round_results: List[SMNLESingleRoundTrainResults]
    posterior_samples: np.ndarray
    posterior_log_prob: Optional[DoublyIntractableLogDensity]
    config: SMNLEConfig = None


class SMNLETrainEvalResults(NamedTuple):
    train_results: SMNLETrainResults
    eval_results: Optional[Any]


def _get_lower_bounds(dist: tdist.Distribution) -> Union[torch.Tensor, None]:
    if isinstance(dist, tdist.Independent):
        base_dist = dist.base_dist
    else:
        base_dist = dist

    if isinstance(base_dist, (tdist.MultivariateNormal, tdist.Normal)):
        return None

    elif isinstance(base_dist, tdist.Uniform):
        if hasattr(base_dist.support, "lower"):
            return torch.ones(dist.event_shape) * base_dist.support.lower
        elif hasattr(base_dist.support, "lower_bound"):
            return torch.ones(dist.event_shape) * base_dist.support.lower_bound
        else:
            return None

    elif isinstance(base_dist, tdist.LogNormal):
        return torch.zeros(dist.event_shape)  # type: ignore

    elif isinstance(base_dist, tdist.TransformedDistribution):
        if hasattr(base_dist.support, "lower"):
            assert base_dist.support.lower.shape == dist.event_shape
            return base_dist.support.lower.shape
        elif hasattr(base_dist.support, "lower_bound"):
            assert base_dist.support.lower_bound.shape == dist.event_shape
            return base_dist.support.lower_bound.shape
        else:
            return None

    else:
        raise ValueError(base_dist)


def _get_upper_bounds(dist: tdist.Distribution) -> Union[torch.Tensor, None]:
    if isinstance(dist, tdist.Independent):
        base_dist = dist.base_dist
    else:
        base_dist = dist

    if isinstance(base_dist, (tdist.MultivariateNormal, tdist.LogNormal, tdist.Normal)):
        return None

    elif isinstance(base_dist, tdist.Uniform):
        if hasattr(base_dist.support, "upper"):
            return torch.ones(dist.event_shape) * base_dist.support.upper
        elif hasattr(base_dist.support, "upper_bound"):
            return torch.ones(dist.event_shape) * base_dist.support.upper_bound
        else:
            return None

    elif isinstance(base_dist, tdist.TransformedDistribution):
        if hasattr(base_dist.support, "upper"):
            assert base_dist.support.upper.shape == dist.event_shape
            return base_dist.support.upper.shape
        elif hasattr(base_dist.support, "upper_bound"):
            assert base_dist.support.upper_bound.shape == dist.event_shape
            return base_dist.support.upper_bound.shape
        else:
            return None

    else:
        raise ValueError


def maybe_reshape(x):
    import jax.numpy as jnp
    if len(x.shape) >=3:
        return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    elif len(x.shape) >= 2:
        return jnp.reshape(x, (x.shape[0] * x.shape[1],))
    else:
        raise ValueError("Can't reshape")



class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self._transform = None


    def fit(self, x: torch.Tensor):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-8

        from torch.distributions.transforms import AffineTransform
        self._transform = AffineTransform(loc=self.mean, scale=self.std).inv


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._transform is not None, "Must fit before transforming"
        ret = self._transform(x)
        assert ret is not None
        return ret



def _sample_savm_jax(prior, net_data_SM, net_theta_SM, thinning_factor, num_chains, burnin_exchange_MCMC, num_mcmc_samples, observation, theta0_vals, x0_vals, aux_MCMC_inner_steps_exchange_MCMC, aux_MCMC_proposal_size_exchange_MCMC, key: PRNGKeyArray) -> torch.Tensor:
    from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist
    import jax.numpy as jnp

    # z-scored space
    from sbi_ebm.sbibm.jax_torch_interop import make_jax_likelihood
    jax_log_likelihood = make_jax_likelihood(net_data_SM, net_theta_SM)
    jax_prior = convert_dist(prior, "numpyro")

    jax_posterior_log_prob = DoublyIntractableLogDensity(
        log_prior=maybe_wrap(lambda x: jax_prior.log_prob(x)),
        log_likelihood=jax_log_likelihood,
        x_obs=jnp.array(observation)[0]  # type: ignore
    )
    
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=SAVMKernelFactory(config=SAVMConfig(
                aux_var_kernel_factory=MALAKernelFactory(MALAConfig(0.1)),
                # aux_var_num_inner_steps=500,
                aux_var_num_inner_steps=aux_MCMC_inner_steps_exchange_MCMC,
                base_var_kernel_factory=RWKernelFactory(config=RWConfig(0.1, jnp.ones((jax_prior.event_shape[0],)))),
                aux_var_init_strategy="x_obs",
            )),
            num_samples=num_mcmc_samples,
            num_chains=num_chains,
            thinning_factor=thinning_factor,
            target_accept_rate=0.5,
            num_warmup_steps=burnin_exchange_MCMC,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True
        )
    )


    alg = config.build_algorithm(jax_posterior_log_prob)
    # alg = alg.init_from_particles(theta0_vals)
    key, subkey = random.split(key)
    alg = alg.init(subkey, jax_prior)

    key, subkey = random.split(key)
    alg, results = jit(type(alg).run)(alg, subkey)
    posterior_samples_torch = torch.from_numpy(np.array(results.samples.xs)).float()
    return posterior_samples_torch


def sample_savm_pytorch(prior, net_data_SM, net_theta_SM,   observation,  num_mcmc_samples, thinning_factor, num_chains,  burnin_exchange_MCMC, aux_MCMC_inner_steps_exchange_MCMC, tuning_window_exchange_MCMC, propose_new_theta_exchange_MCMC, proposal_size_exchange_MCMC, aux_MCMC_proposal_size_exchange_MCMC, bridging_exch_MCMC,  seed, debug_level,) -> torch.Tensor:
    from smnle.src.exchange_mcmc import MIN_NUM_CHAINS

    if num_chains > MIN_NUM_CHAINS:
        initial_theta_exchange_MCMC = cast(
            torch.Tensor, prior.sample(sample_shape=torch.Size((num_chains,)))
        ).numpy()
    else:
        initial_theta_exchange_MCMC = cast(torch.Tensor, prior.sample()).numpy()
    # initial_theta_exchange_MCMC = cast(torch.Tensor, torch.zeros(task.dim_parameters)).numpy()


    assert propose_new_theta_exchange_MCMC in (
        "transformation",
        "adaptive_transformation",
        "norm",
    )
    proposal_size_exchange_MCMC_arr = proposal_size_exchange_MCMC * np.ones_like(
        initial_theta_exchange_MCMC
    )

    x_obs = observation.numpy()

    start = time()


    # should work for numpy and torch
    def _prior(theta):
        if isinstance(theta, torch.Tensor):
            return prior.log_prob(theta).exp()
        elif isinstance(theta, np.ndarray):
            return (
                prior.log_prob(torch.from_numpy(theta).float())
                .exp()
                .detach()
                .numpy()
            )
        else:
            raise ValueError("theta must be either torch.Tensor or np.ndarray")


    _T = num_mcmc_samples * thinning_factor / num_chains
    assert int(_T) == _T
    _T = int(_T)

    trace_exchange = exchange_MCMC_with_SM_statistics(
        x_obs,
        initial_theta_exchange_MCMC,
        # lambda x: uniform_prior_theta(x, lower_bounds,
        #                               upper_bounds),
        _prior,
        net_data_SM,
        net_theta_SM,
        None,  # scalers
        None,  # scalers
        propose_new_theta_exchange_MCMC,
        T=_T,
        burn_in=burnin_exchange_MCMC,
        tuning_window_size=tuning_window_exchange_MCMC,
        aux_MCMC_inner_steps=aux_MCMC_inner_steps_exchange_MCMC,
        aux_MCMC_proposal_size=aux_MCMC_proposal_size_exchange_MCMC,
        K=bridging_exch_MCMC,
        seed=seed,
        debug_level=debug_level,
        lower_bounds_theta=None,
        upper_bounds_theta=None,
        sigma=proposal_size_exchange_MCMC_arr,
        num_chains=num_chains,
        thinning_factor=thinning_factor,
    )

    trace_exchange_burned_in = trace_exchange[burnin_exchange_MCMC:]

    trace_exchange_burned_in = trace_exchange_burned_in[::thinning_factor]
    trace_exchange_burned_in = trace_exchange_burned_in.reshape(
        num_mcmc_samples, initial_theta_exchange_MCMC.shape[-1]
    )
    posterior_samples = torch.from_numpy(trace_exchange_burned_in).float()
    return posterior_samples



def combine(new_theta_vect, new_samples_matrix, theta_vect, theta_vect_test, samples_matrix, samples_matrix_test, this_round_num_test_simulations):
    assert isinstance(new_samples_matrix, torch.Tensor)

    new_theta_vect_test = new_theta_vect[:this_round_num_test_simulations]

    new_samples_matrix_test = new_samples_matrix[
        :this_round_num_test_simulations
    ]

    new_theta_vect = new_theta_vect[this_round_num_test_simulations:]
    new_samples_matrix = new_samples_matrix[this_round_num_test_simulations:]

    assert isinstance(theta_vect_test, torch.Tensor)
    theta_vect = torch.cat((theta_vect, new_theta_vect))
    theta_vect_test = torch.cat((theta_vect_test, new_theta_vect_test))

    # concat samples_matrix and new_samples_matrix using numpy
    samples_matrix = torch.cat((samples_matrix, new_samples_matrix))
    samples_matrix_test = torch.cat(
        (samples_matrix_test, new_samples_matrix_test)
    )
    return theta_vect, theta_vect_test, samples_matrix, samples_matrix_test


def smnle(
    # sbibm
    task: Union[Task, str],
    num_observation: Optional[int] = None,
    num_rounds: int = 1,
    observation: Optional[torch.Tensor] = None,
    num_simulations: int = 1000,
    automatic_transforms_enabled: bool = True,
    num_samples: int = 1000,
    # SM training args
    technique: Literal["SM", "SSM"] = "SM",
    epochs: int = 500,
    no_scheduler: bool = False,
    noise_sliced: str = "radermacher",
    no_var_red_sliced: bool = False,
    no_bn: bool = False,
    affine_batch_norm: bool = False,
    lr_data: float = 0.001,
    SM_lr_theta: float = 0.001,
    batch_size: int = 1000,
    no_early_stop: bool = False,
    update_batchnorm_running_means_before_eval: bool = False,
    momentum: float = 0.9,
    epochs_before_early_stopping: int = 200,
    epochs_test_interval: int = 10,
    scale_samples: bool = True,
    scale_parameters: bool = True,
    # save_net_at_each_epoch: bool = False,
    seed: int = 42,
    cuda: bool = False,
    lam: int = 0,
    # inference args:
    num_chains: int = 100,
    thinning_factor: int = 10,
    propose_new_theta_exchange_MCMC: Literal[
        "transformation", "adaptive_transformation", "norm"
    ] = "transformation",
    burnin_exchange_MCMC: int = 300,
    tuning_window_exchange_MCMC: int = 100,
    aux_MCMC_inner_steps_exchange_MCMC: int = 100,
    aux_MCMC_proposal_size_exchange_MCMC: float = 0.1,
    proposal_size_exchange_MCMC: float = 0.1,
    bridging_exch_MCMC: int = 0,
    # Misc
    debug_level: int = 100,
    theta_vect: Optional[torch.Tensor] = None,
    use_jax_mcmc: bool = False,
    use_data_from_past_rounds: bool = True,
    evaluate_posterior: bool = False,
    use_tqdm: bool = False,
):
    torch.manual_seed(seed)
    key = random.PRNGKey(seed)

    start = time()

    if isinstance(task, str):
        from sbi_ebm.sbibm.tasks import get_task
        task = get_task(task)

    # DATA:
    if observation is None:
        assert num_observation is not None
        observation = task.get_observation(num_observation)
    else:
        assert num_observation is None
        # assert len(observation.shape) == 1

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    # NETWORKS
    var_red_sliced = not no_var_red_sliced
    batch_norm_last_layer = not no_bn

    # FP_lr = lr_data
    SM_lr = lr_data
    early_stopping = not no_early_stop

    if SM_lr is None:
        SM_lr = 0.001
    if SM_lr_theta is None:
        SM_lr_theta = 0.001

    assert num_simulations > 0

    this_round_num_simulations = num_simulations // num_rounds

    this_round_num_train_simulations = max(int(this_round_num_simulations * 0.9), 1)
    this_round_num_test_simulations = (
        this_round_num_simulations - this_round_num_train_simulations
    )

    assert this_round_num_train_simulations > 0
    assert this_round_num_test_simulations > 0

    # no need to unconstrain params?
    # param_transforms = task._get_transforms(False)["parameters"]
    param_transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]

    model = task._get_pyro_model(posterior=False)
    from sbibm.utils.pyro import get_log_prob_fn

    _, transforms_dct = get_log_prob_fn(model, automatic_transform_enabled=automatic_transforms_enabled)

    if "data" in transforms_dct:
        data_transforms = transforms_dct["data"]
    else:
        data_transforms = pyro_transforms.identity_transform


    # param_transforms = pyro_transforms.SigmoidTransform()
    # data_transforms = pyro_transforms.SigmoidTransform()

    print("data_transforms", data_transforms)
    print("param_transforms", param_transforms)

    prior = wrap_prior_dist(prior, param_transforms)
    simulator = wrap_simulator_fn(simulator, param_transforms, data_transforms)
    observation = data_transforms(observation)
    assert observation is not None

    if theta_vect is None:
        theta_vect = prior.sample(torch.Size((this_round_num_train_simulations,)))
        theta_vect_test = prior.sample(torch.Size((this_round_num_test_simulations,)))

        samples_matrix = simulator(theta_vect).float()
        samples_matrix_test = simulator(theta_vect_test).float()
    else:
        theta_vect = param_transforms(theta_vect)
        samples_matrix = simulator(theta_vect).float()
        assert theta_vect is not None

        theta_vect_test = theta_vect[-this_round_num_test_simulations:]
        theta_vect = theta_vect[:-this_round_num_test_simulations]

        samples_matrix_test = samples_matrix[-this_round_num_test_simulations:]
        samples_matrix = samples_matrix[:-this_round_num_test_simulations]

    single_round_results = []
    trace_exchange_burned_in = None

    jax_posterior_log_prob = None
    jax_log_likelihood = None
    for round_no in range(num_rounds):
        assert theta_vect is not None
        assert theta_vect_test is not None
        assert samples_matrix is not None
        assert samples_matrix_test is not None
        assert isinstance(samples_matrix, torch.Tensor)
        assert isinstance(samples_matrix_test, torch.Tensor)
        assert isinstance(theta_vect, torch.Tensor)
        assert isinstance(theta_vect_test, torch.Tensor)


        if scale_parameters:
            scaler_theta = TorchStandardScaler()
            scaler_theta.fit(theta_vect)

            theta_vect_transformed = scaler_theta.transform(theta_vect)
            theta_vect_test_transformed = scaler_theta.transform(theta_vect_test)
            assert scaler_theta._transform is not None
            scaler_theta_transform = scaler_theta._transform
        else:
            from torch.distributions.transforms import identity_transform
            scaler_theta_transform = identity_transform
            theta_vect_transformed = theta_vect
            theta_vect_test_transformed = theta_vect_test


        scaled_prior = wrap_prior_dist(prior, scaler_theta_transform)


        if scale_samples:
            scaler_samples = TorchStandardScaler()
            scaler_samples.fit(samples_matrix)
            samples_matrix_transformed = scaler_samples.transform(samples_matrix)
            samples_matrix_test_transformed = scaler_samples.transform(samples_matrix_test)
            scaled_observation = scaler_samples.transform(observation)
            assert scaler_samples._transform is not None
            scaler_samples_transform = scaler_samples._transform
        else:
            from torch.distributions.transforms import identity_transform
            scaler_samples_transform = identity_transform
            samples_matrix_transformed = samples_matrix
            samples_matrix_test_transformed = samples_matrix_test
            scaled_observation = observation

        print("Data generation took {:.4f} seconds".format(time() - start))

        nonlinearity = torch.nn.Softplus

        net_data_SM_architecture = createDefaultNNWithDerivatives(
            task.dim_data, 50, [50, 50, 50], nonlinearity=nonlinearity
        )
        net_theta_SM_architecture = createDefaultNN(
            task.dim_parameters,
            49,
            [50, 50, 50],
            nonlinearity=nonlinearity(),
            batch_norm_last_layer=batch_norm_last_layer,
            affine_batch_norm=affine_batch_norm,
            batch_norm_last_layer_momentum=momentum,
        )

        net_data_SM = net_data_SM_architecture()
        net_theta_SM = net_theta_SM_architecture()

        statistics_learning = ExpFamStatistics(
            model=None,
            statistics_calc=Identity(),
            backend=BackendDummy(),  # backend and model are not used
            simulations_net=net_data_SM,
            parameters_net=net_theta_SM,
            parameters=theta_vect_transformed.detach().clone().numpy(),
            simulations=samples_matrix_transformed.detach().clone().numpy(),
            parameters_val=theta_vect_test_transformed.detach().clone().numpy(),
            simulations_val=samples_matrix_test_transformed.detach().clone().numpy(),
            scale_samples=False,
            scale_parameters=False,
            lower_bound_simulations=None,
            upper_bound_simulations=None,
            sliced=technique == "SSM",
            noise_type=noise_sliced,
            variance_reduction=var_red_sliced and not noise_sliced == "sphere",
            n_epochs=epochs,
            batch_size=batch_size,
            lr_simulations=SM_lr,
            lr_parameters=SM_lr_theta,
            seed=seed,
            start_epoch_early_stopping=epochs_before_early_stopping,
            epochs_early_stopping_interval=epochs_test_interval,
            early_stopping=early_stopping,
            scheduler_parameters=False if no_scheduler else None,
            scheduler_simulations=False if no_scheduler else None,
            cuda=cuda,
            lam=lam,
            batch_norm_update_before_test=update_batchnorm_running_means_before_eval,
            use_tqdm=use_tqdm
        )

        if round_no == num_rounds - 1:
            num_mcmc_samples = num_samples
        else:
            num_mcmc_samples = num_simulations // num_rounds

        if use_jax_mcmc:
            key, subkey = random.split(key)
            posterior_samples = _sample_savm_jax(
                scaled_prior, net_data_SM, net_theta_SM, thinning_factor, num_chains, burnin_exchange_MCMC, num_mcmc_samples, scaled_observation,
                theta_vect_transformed[:num_mcmc_samples], samples_matrix_transformed[:num_mcmc_samples], aux_MCMC_inner_steps_exchange_MCMC, aux_MCMC_proposal_size_exchange_MCMC,
                subkey
            )

        else:
            posterior_samples = sample_savm_pytorch(
                scaled_prior, net_data_SM, net_theta_SM,   scaled_observation,  num_mcmc_samples, thinning_factor, num_chains,  burnin_exchange_MCMC,
                aux_MCMC_inner_steps_exchange_MCMC, tuning_window_exchange_MCMC, propose_new_theta_exchange_MCMC, proposal_size_exchange_MCMC, aux_MCMC_proposal_size_exchange_MCMC,
                bridging_exch_MCMC,  seed, debug_level
            )

        posterior_samples = scaler_theta_transform.inv(posterior_samples)
        assert isinstance(posterior_samples, torch.Tensor)


        from sbi_ebm.sbibm.jax_torch_interop import make_jax_likelihood
        jax_log_likelihood = make_jax_likelihood(
            net_data_SM, net_theta_SM,
            pyro_transforms.ComposeTransform([data_transforms, scaler_samples_transform]),
            pyro_transforms.ComposeTransform([param_transforms, scaler_theta_transform]),
        )

        single_round_result = SMNLESingleRoundTrainResults(
            statistics=statistics_learning,
            net_data=net_data_SM,
            net_theta=net_theta_SM,
            scaler_data=None,  # type: ignore
            scaler_theta=None,  # type: ignore
            posterior_samples=param_transforms.inv(posterior_samples),
            train_theta=theta_vect,
            train_data=samples_matrix,
            jax_log_likelihood=jax_log_likelihood,
        )

        single_round_results.append(single_round_result)

        if round_no < num_rounds - 1:
            # assert posterior_samples is not None
            # assert isinstance(posterior_samples, torch.Tensor)
            # new_theta_vect = posterior_samples
            # new_theta_vect = new_theta_vect[np.random.permutation(len(new_theta_vect))]

            new_samples_matrix = simulator(posterior_samples).float()
            assert isinstance(new_samples_matrix, torch.Tensor)
            if use_data_from_past_rounds:
                print('combining with past round data')
                theta_vect, theta_vect_test, samples_matrix, samples_matrix_test = combine(
                    posterior_samples, new_samples_matrix, theta_vect, theta_vect_test, samples_matrix, samples_matrix_test,
                    this_round_num_test_simulations
                )
                print(len(theta_vect))
            else:
                samples_matrix = new_samples_matrix
                theta_vect = posterior_samples

                theta_vect_test = theta_vect[-this_round_num_test_simulations:]
                theta_vect = theta_vect[:-this_round_num_test_simulations]

                samples_matrix_test = samples_matrix[-this_round_num_test_simulations:]
                samples_matrix = samples_matrix[:-this_round_num_test_simulations]


    from sbi_ebm.sbibm.jax_torch_interop import make_jax_likelihood

    final_posterior_samples = single_round_results[-1].posterior_samples


    task_config = TaskConfig(
        simulator, convert_dist(prior, implementation="numpyro"),
        jnp.array(observation), task.name, num_observation if num_observation is not None else 0, use_calibration_kernel=False
    )
    config = SMNLEConfig(
        task=task_config,
        num_observation=num_observation,
        num_rounds=num_rounds,
        observation=observation,
        num_simulations=num_simulations,
        automatic_transforms_enabled=automatic_transforms_enabled,
        num_samples=num_samples,
        technique=technique,
        epochs=epochs,
        no_scheduler=no_scheduler,
        noise_sliced=noise_sliced,
        no_var_red_sliced=no_var_red_sliced,
        no_bn=no_bn,
        affine_batch_norm=affine_batch_norm,
        lr_data=lr_data,
        SM_lr_theta=SM_lr_theta,
        batch_size=batch_size,
        no_early_stop=no_early_stop,
        update_batchnorm_running_means_before_eval=update_batchnorm_running_means_before_eval,
        momentum=momentum,
        epochs_before_early_stopping=epochs_before_early_stopping,
        epochs_test_interval=epochs_test_interval,
        scale_samples=scale_samples,
        scale_parameters=scale_parameters,
        seed=seed,
        cuda=cuda,
        lam=lam,
        num_chains=num_chains,
        thinning_factor=thinning_factor,
        propose_new_theta_exchange_MCMC=propose_new_theta_exchange_MCMC,
        burnin_exchange_MCMC=burnin_exchange_MCMC,
        tuning_window_exchange_MCMC=tuning_window_exchange_MCMC,
        aux_MCMC_inner_steps_exchange_MCMC=aux_MCMC_inner_steps_exchange_MCMC,
        aux_MCMC_proposal_size_exchange_MCMC=aux_MCMC_proposal_size_exchange_MCMC,
        proposal_size_exchange_MCMC=proposal_size_exchange_MCMC,
        bridging_exch_MCMC=bridging_exch_MCMC,
        debug_level=debug_level,
        theta_vect=theta_vect,
        use_jax_mcmc=use_jax_mcmc,
        use_data_from_past_rounds=use_data_from_past_rounds,
        evaluate_posterior=evaluate_posterior,
    )

    train_results = SMNLETrainResults(
        single_round_results=single_round_results,
        posterior_samples=final_posterior_samples,
        posterior_log_prob=jax_posterior_log_prob,
        config=config,
    )
    # means_exchange = np.mean(trace_exchange_burned_in, axis=0)
    # trace_exchange_subsample = subsample_trace(trace_exchange_burned_in, size=subsample_size)  # used to compute wass dist
    if evaluate_posterior:
        assert num_observation is not None
        eval_results = _evaluate_posterior(
            train_results.posterior_samples, task, num_observation
        )
    else:
        eval_results = None


    return SMNLETrainEvalResults(train_results, eval_results)
