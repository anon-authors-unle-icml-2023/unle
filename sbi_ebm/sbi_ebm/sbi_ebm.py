import time
import os
import copy
import copyreg
from typing import Callable, Generic, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union, cast, Any

# from jax.config import config  # type: ignore
# config.update("jax_debug_nans", True)

import jax
from jax.lax import fori_loop  # type: ignore
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import random, vmap, jit
from jax.tree_util import tree_map, tree_leaves
from jax.nn import logsumexp
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from numpyro.distributions.constraints import _Real
from typing_extensions import Self, TypeGuard
from sbi_ebm.calibration.calibration import CalibrationMLP

from sbi_ebm.distributions import BlockDistribution, ThetaConditionalLogDensity, maybe_wrap, maybe_wrap_log_l, DoublyIntractableLogDensity
from sbi_ebm.mog import MOGTrainingConfig, fit_mog
from sbi_ebm.pytypes import Array, LogDensity_T, Numeric, PRNGKeyArray, PyTreeNode, Simulator_T
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation
from sbi_ebm.samplers.inference_algorithms.base import IAC_T, InferenceAlgorithmConfig, InferenceAlgorithmFactory, PA_T_co
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm, MCMCAlgorithmFactory, MCMCConfig, MCMCResults

from sbi_ebm.sbibm.tasks import JaxTask, ReferencePosterior

from .data import SBIDataset, SBIParticles, ZScorer
from .likelihood_ebm import (LD_T, LikelihoodFactory, Trainer, TrainerResults,
                             TrainingConfig, TrainState)
from .samplers.inference_algorithms.importance_sampling.smc import SMCConfig, SMC, SMCFactory
from sbi_ebm.likelihood_ebm import _EBMRatio

# https://github.com/pyro-ppl/numpyro/issues/1378
copyreg.pickle(_Real, lambda _: "real")


import jax.numpy as jnp


def get_total_transform(z_scorer):
    def inverse_affine_transform(t):
        from numpyro.distributions.transforms import AffineTransform
        return AffineTransform(scale=1/t.scale, loc=-t.loc/t.scale)

    def concat_affine_transform(t1, t2):
        from numpyro.distributions.transforms import AffineTransform
        return AffineTransform(scale=jnp.concatenate([t1.scale, t2.scale]), loc=jnp.concatenate([t1.loc, t2.loc]))

    params_forward_transform = z_scorer.get_transform("params").parts[0].inv
    obs_forward_transform = z_scorer.get_transform("observations").parts[0].inv

    return concat_affine_transform(
        inverse_affine_transform(params_forward_transform), inverse_affine_transform(
            obs_forward_transform)
    )


def adapt_for_new_scaling(z_transform_prev, z_transform_new, params):
    # z_transform_prev: orig space to z-scored space
    # z_transform_new: orig space to z-scored space
    # params: params in z-scored space

    def inverse(scale, bias):
        return 1 / scale, - bias / scale

    def compose_componentwise_transform(outer_scale, outer_bias, inner_scale, inner_bias):
        return outer_scale * inner_scale, outer_scale * inner_bias + outer_bias

    def compose_dense_and_componentwise_transform(outer_scale, outer_bias, inner_scale, inner_bias):
        return jnp.diag(inner_scale) @ outer_scale, inner_bias @ outer_scale + outer_bias

    outermost_scale = params['params']['energy_network']['layers_0']['kernel']
    outermost_bias = params['params']['energy_network']['layers_0']['bias']

    new_kernel, new_bias = compose_dense_and_componentwise_transform(
        outermost_scale, outermost_bias,
        *compose_componentwise_transform(
            z_transform_prev.scale, z_transform_prev.loc,
            *inverse(z_transform_new.scale, z_transform_new.loc)
        )
    )
    new_params = params.unfreeze().copy()

    new_params['params']['energy_network']['layers_0']['kernel'] = new_kernel
    new_params['params']['energy_network']['layers_0']['bias'] = new_bias

    from flax.core import FrozenDict
    return FrozenDict(new_params)


def rescale_params(prev_params, z_scorer_prev, z_scorer_new):
    new_z_transform = get_total_transform(z_scorer_new)
    prev_z_transform = get_total_transform(z_scorer_prev)
    return adapt_for_new_scaling(prev_z_transform, new_z_transform, prev_params)


def rescale_zscored_sample(prev_zscored_sample, z_scorer_prev: ZScorer, z_scorer_new: ZScorer):
    inverse_prev_transform = z_scorer_prev.get_transform("params").inv
    new_transform = z_scorer_new.get_transform("params")
    return new_transform(inverse_prev_transform(prev_zscored_sample))

def get_zscoring_ratio(z_scorer_prev: ZScorer, z_scorer_new: ZScorer):
    inverse_prev_transform = z_scorer_prev.get_transform("params").parts[0].inv
    inverse_new_transform = z_scorer_new.get_transform("params").parts[0].inv

    assert isinstance(inverse_prev_transform, np_transforms.AffineTransform)
    assert isinstance(inverse_new_transform, np_transforms.AffineTransform)

    return jnp.sqrt(jnp.average(inverse_prev_transform.scale / inverse_new_transform.scale))

class SBIEBMCalibrationNet(struct.PyTreeNode):
    params: PyTreeNode

    def log_prob(self, param):
        from sbi_ebm.calibration.calibration import CalibrationMLP
        logits = CalibrationMLP().apply({"params": self.params}, param)
        return jax.nn.log_softmax(logits)[..., 1]


class EBMPosterior(np_distributions.Distribution):
    arg_constraints = {"likelihood_factory": None, "prior": None, "x": None}
    meta_fields = {"z_transform": None, "x_transform": None}

    def __init__(
        self,
        prior: np_distributions.Distribution,
        x: Array,
        z_transform: np_transforms.Transform = np_transforms.IdentityTransform(),
        x_transform: np_transforms.Transform = np_transforms.IdentityTransform(),
        calibration_net: Optional[Union[SBIEBMCalibrationNet, Callable]] = None,
        likelihood_factory: Optional[LikelihoodFactory] = None,
        ratio: Optional[_EBMRatio] = None
    ):
        if likelihood_factory is not None:
            assert ratio is None
            self.ebm_model_type = "likelihood"
        elif ratio is not None:
            assert likelihood_factory is None
            self.ebm_model_type = "ratio"
        else:
            raise ValueError("Cannot specify both likelihood_factory and ratio")

        self.likelihood_factory = likelihood_factory
        self.ratio = ratio

        self.ratio = ratio
        self.x = x

        self.prior = prior  # lives in zscored space
        self.z_transform = z_transform
        self.x_transform = x_transform

        self.support = self.z_transform.codomain

        if calibration_net is None:
            self.calibration_net_log_prob = lambda x: 0.
        else:
            self.calibration_net = calibration_net
            self.calibration_net_log_prob = calibration_net.log_prob

        assert len(prior.event_shape) > 0
        super(EBMPosterior, self).__init__(
            batch_shape=(), event_shape=prior.event_shape
        )
    
    def replace_params(self, params):
        copied_self = copy.copy(self)
        if copied_self.ebm_model_type == "likelihood":
            assert copied_self.likelihood_factory is not None
            new_likelihood_factory = copied_self.likelihood_factory.replace(params=params)
            copied_self.likelihood_factory = new_likelihood_factory
        elif copied_self.ebm_model_type == "ratio":
            assert copied_self.ratio is not None
            new_ratio = copied_self.ratio.replace(params=params)
            copied_self.ratio = new_ratio
        else:
            raise ValueError("Unknown EBM model type")

        return copied_self



    def _log_prob_likelihood_type(self, z: Array) -> Numeric:
        transformed_prior = np_distributions.TransformedDistribution(
            self.prior, self.z_transform
        )
        assert self.likelihood_factory is not None
        transformed_likelihood = np_distributions.TransformedDistribution(
            self.likelihood_factory(self.z_transform.inv(z)),
            self.x_transform,
        )
        return transformed_prior.log_prob(z) + transformed_likelihood.log_prob(self.x) + self.calibration_net_log_prob(self.z_transform.inv(z))


    def _log_prob_ratio_type(self, z: Array) -> Numeric:
        transformed_prior = np_distributions.TransformedDistribution(
            self.prior, self.z_transform
        )
        assert self.ratio is not None
        
        return transformed_prior.log_prob(z) + self.ratio(self.z_transform.inv(z), self.x_transform.inv(self.x)) + self.calibration_net_log_prob(self.z_transform.inv(z))

    def log_prob(self, z) -> Numeric:
        if not hasattr(self, "ebm_model_type"):
            return self._log_prob_likelihood_type(z)
        if self.ebm_model_type == "likelihood":
            return self._log_prob_likelihood_type(z)
        elif self.ebm_model_type == "ratio":
            return self._log_prob_ratio_type(z)
        else:
            raise ValueError("Unknown EBM model type")

    def _log_prob_zscored_space_likelihood_type(self, z):
        # self.x is assumed to live in the untransformed space, hence the transform call
        assert self.likelihood_factory is not None
        return self.prior.log_prob(z) + self.likelihood_factory(z).log_prob(
            self.x_transform.inv(self.x)
        ) + self.calibration_net_log_prob(z)

    def _log_prob_zscored_space_ratio_type(self, z: Array) -> Numeric:
        assert self.ratio is not None
        return self.prior.log_prob(z) + self.ratio(z, self.x_transform.inv(self.x)) + self.calibration_net_log_prob(z)


    def log_prob_zscored_space(self, z):
        if not hasattr(self, "ebm_model_type"):
            return self._log_prob_zscored_space_likelihood_type(z)
        elif self.ebm_model_type == "likelihood":
            return self._log_prob_zscored_space_likelihood_type(z)
        elif self.ebm_model_type == "ratio":
            return self._log_prob_zscored_space_ratio_type(z)
        else:
            raise ValueError("Unknown EBM model type")


    def _joint_log_prob_likelihood_type(self, z: Array, x: Array) -> Numeric:
        assert self.likelihood_factory is not None
        transformed_prior = np_distributions.TransformedDistribution(
            self.prior, self.z_transform
        )
        transformed_likelihood = np_distributions.TransformedDistribution(
            self.likelihood_factory(self.z_transform.inv(z)),
            self.x_transform,
        )
        return transformed_prior.log_prob(z) + transformed_likelihood.log_prob(x)

    def _joint_log_prob_zscored_space(self, z: Array, x: Array) -> Numeric:
        assert self.likelihood_factory is not None
        return self.prior.log_prob(z) + self.likelihood_factory(z).log_prob(x)

    def _log_likelihood(self, z: Array, x: Array) -> Numeric:
        assert self.likelihood_factory is not None
        transformed_likelihood = np_distributions.TransformedDistribution(
            self.likelihood_factory(self.z_transform.inv(z)),
            self.x_transform,
        )
        return transformed_likelihood.log_prob(x)

    def _log_likelihood_zscored_space(self, z: Array, x: Array) -> Numeric:
        assert self.likelihood_factory is not None
        return self.likelihood_factory(z).log_prob(x)

    def _prior(self, z: Array) -> Numeric:
        transformed_prior = np_distributions.TransformedDistribution(
            self.prior, self.z_transform
        )
        return transformed_prior.log_prob(z)

    def _prior_zscored_space(self, z: Array) -> Numeric:
        return self.prior.log_prob(z)


class KDEApproxDist(np_distributions.Distribution):
    arg_contraints = {"log_prob": None}

    def __init__(self, dist: np_distributions.Distribution, smc_proposal:
                 np_distributions.Distribution, key: PRNGKeyArray,
                 mog_config: MOGTrainingConfig,
                 fit_in_z_score_space: bool = False,
                 train_samples: Optional[Array] = None, t_prior: float = 0.1):
        self._dist = dist
        self._smc_proposal = smc_proposal
        self._prior = smc_proposal
        self._fit_in_zscore_space = fit_in_z_score_space

        self._smc_config = SMCConfig(
            num_samples=1000,
            num_steps=5000, ess_threshold=0.8,
            inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
            inner_kernel_steps=5,
            record_trajectory=False
        )
        self.support = self._dist.support
        unconstraining_transform = np_transforms.biject_to(dist.support).inv

        if fit_in_z_score_space:
            self._z_scored_dist = np_distributions.TransformedDistribution(
                dist, unconstraining_transform
            )
        else:
            self._z_scored_dist = dist

        if train_samples is None:
            self._train_samples = self._get_orig_dist_samples(
                key, sample_shape=(10000,), fit_in_zscore_space=fit_in_z_score_space
            )
        else:
            if fit_in_z_score_space:
                self._train_samples = unconstraining_transform(train_samples)
            else:
                self._train_samples = train_samples

        self._mog_config = mog_config

        key, subkey = random.split(key)
        self._mog = fit_mog(self._train_samples, self._mog_config, key=subkey).to_dist()

        self._t_prior = t_prior

        if fit_in_z_score_space:
            self._mog = np_distributions.TransformedDistribution(
                self._mog, np_transforms.biject_to(dist.support)
            )

        super(KDEApproxDist, self).__init__(
            batch_shape=self._dist.batch_shape, event_shape=self._dist.event_shape
        )

    def log_prob(self, z: Array) -> Numeric:
        tempered_log_prob = self._mog.log_prob(z)
        return jax.nn.logsumexp(
            jnp.array([tempered_log_prob, self._smc_proposal.log_prob(z)]),
            b=jnp.array([1 - self._t_prior, self._t_prior])
        )
        # return self._log_prob(z)

    def _get_orig_dist_samples(self, key, sample_shape=(), fit_in_zscore_space: bool = False) -> Array:
        if fit_in_zscore_space:
            unconstrained_log_prob = self._z_scored_dist.log_prob
        else:
            unconstrained_log_prob = self._dist.log_prob

        alg = SMC(config=self._smc_config.replace(num_samples=sample_shape[0]), log_prob=maybe_wrap(lambda x: unconstrained_log_prob(x)))
        key, subkey = random.split(key)
        alg = alg.init(key=subkey, dist=self._smc_proposal)

        key, subkey = random.split(key)
        alg, results = alg.run(subkey)

        key, key_resampling = random.split(key)
        samples = results.samples.resample_and_reset_weights(key_resampling)
        # samples = inv_transform(samples.xs)
        samples = samples.xs
        return samples

    def sample(self, key: PRNGKeyArray, sample_shape: tuple = ()) -> Array:
        key, subkey = random.split(key)
        prior_samples = self._prior.sample(subkey, sample_shape)

        key, subkey = random.split(key)
        mog_samples = self._mog.sample(subkey, sample_shape)

        key, subkey = random.split(key)
        use_prior = np_distributions.Bernoulli(jnp.array([self._t_prior])).sample(
            subkey, sample_shape
        )
        return prior_samples * use_prior + mog_samples * (1 - use_prior)


class TaskConfig(struct.PyTreeNode):
    simulator: Simulator_T = struct.field(pytree_node=False)
    prior: np_distributions.Distribution = struct.field(pytree_node=True)
    x_obs: Array = struct.field(pytree_node=True)
    task_name: str = struct.field(pytree_node=False, default="")
    num_observation: int = struct.field(pytree_node=False, default=0)
    use_calibration_kernel: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def from_task(cls, task: JaxTask, num_observation: int, use_calibration_kernel: bool = False) -> Self:
        return cls(
            simulator=task.get_simulator(),
            prior=task.get_prior_dist(),
            x_obs=task.get_observation(num_observation)[0],
            task_name=task.task.name,
            num_observation=num_observation,
            use_calibration_kernel=use_calibration_kernel
        )

    @property
    def dim_data(self) -> int:
        return self.x_obs.shape[0]

    @property
    def dim_parameters(self) -> int:
        return self.prior.event_shape[0]

    def __reduce__(self):
        from sbi_ebm.sbibm.tasks import get_task
        t = JaxTask(get_task(self.task_name))
        return TaskConfig.from_task, (t, self.num_observation)


IAC_T_co = TypeVar("IAC_T_co", bound=InferenceAlgorithmConfig, covariant=True)

class InferenceConfig(struct.PyTreeNode):
    num_samples: int
    sampling_config: InferenceAlgorithmFactory
    sampling_init_dist: np_distributions.Distribution
    should_z_score: bool = struct.field(pytree_node=False, default=False)


class ProposalConfig(struct.PyTreeNode):
    init_proposal: np_distributions.Distribution = struct.field(pytree_node=False)
    tempering_enabled: bool = struct.field(pytree_node=False)
    mog_config: Optional[MOGTrainingConfig] = struct.field(pytree_node=True, default=None)
    t_prior: float = struct.field(pytree_node=False, default=0.1)


class PreProcessingConfig(struct.PyTreeNode):
    normalize: bool = struct.field(pytree_node=False, default=True)
    biject_to_unconstrained_space: bool = struct.field(pytree_node=False, default=False)


class CheckpointConfig(struct.PyTreeNode):
    should_start_from_checkpoint: bool = struct.field(pytree_node=False, default=False)
    init_checkpoint_path: str = struct.field(pytree_node=False, default="")
    init_checkpoint_round: int = struct.field(pytree_node=False, default=-1)
    checkpoint_path: str = struct.field(pytree_node=False, default="")
    should_checkpoint: bool = struct.field(pytree_node=False, default=False)
    single_round_results: Optional[List["SingleRoundResults"]] = struct.field(pytree_node=True, default=None)


class Config(struct.PyTreeNode):
    num_samples: Tuple[int]
    training: TrainingConfig
    task: TaskConfig
    inference: InferenceConfig
    proposal: ProposalConfig
    frac_test_samples: float = 0.15
    discard_prior_samples: bool = struct.field(pytree_node=False, default=False)
    use_data_from_past_rounds: bool = struct.field(pytree_node=False, default=True)
    preprocessing: PreProcessingConfig = struct.field(pytree_node=True, default=PreProcessingConfig())
    checkpointing: CheckpointConfig = struct.field(pytree_node=True, default=CheckpointConfig())
    n_sigma: float = struct.field(pytree_node=False, default=3.0)


def _z_score_proposal_and_data(dataset: SBIDataset, normalize: bool, biject_to_unconstrained_space: bool) -> Tuple[SBIDataset, ZScorer]:
    z_scorer = ZScorer.create_and_fit(dataset.train_samples, normalize=normalize, biject_to_unconstrained_space=biject_to_unconstrained_space)

    z_scored_dataset = SBIDataset(
        train_samples=z_scorer.transform(dataset.train_samples),
        test_samples=z_scorer.transform(dataset.test_samples),
    )

    return z_scored_dataset, z_scorer

class SingleRoundResults(NamedTuple):
    dataset: SBIDataset
    config: Config
    posterior: EBMPosterior
    z_scorer: ZScorer
    posterior_samples: Array
    train_results: TrainerResults
    x_obs: Array
    complete_dataset: Optional[SBIParticles] = None  # includes samples with nans
    simulation_time: float = 0.0
    inference_time: float = 0.0

    def get_posterior(self, iter_no: int) -> EBMPosterior:
        assert self.train_results.trajectory is not None
        this_iter_params = tree_map(
            lambda x: x[iter_no], self.train_results.trajectory.params
        )
        copied_posterior = copy.copy(self.posterior)
        if copied_posterior.likelihood_factory is not None:
            copied_posterior.likelihood_factory = (
                copied_posterior.likelihood_factory.replace(params=this_iter_params)
            )
        else:
            assert copied_posterior.ratio is not None
            copied_posterior.ratio = (
                copied_posterior.ratio.replace(params=this_iter_params)
            )

        return copied_posterior

    def get_joint_samples(
        self, iter_nos: List[int], key: PRNGKeyArray, num_samples: int
    ) -> Array:
        assert self.train_results.trajectory is not None
        params = tree_map(
            lambda x: jnp.array(np.array(x)[iter_nos]),
            self.train_results.trajectory.params
        )
        def _get_joint_samples(params, key):

            posterior = copy.copy(self.posterior)
            posterior = posterior.replace_params(params)

            joint_log_prob = maybe_wrap(
                lambda x: posterior._joint_log_prob_zscored_space(
                    x[:self.dataset.dim_params], x[self.dataset.dim_params:]
                )
            )
            proposal = np_distributions.MultivariateNormal(
                jnp.zeros((self.dataset.dim_params + self.dataset.dim_observations),),
                covariance_matrix=jnp.eye(self.dataset.dim_params + self.dataset.dim_observations),
            )
            config = SMCConfig(
                num_samples=1000,
                num_steps=200, ess_threshold=0.8,
                inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
                inner_kernel_steps=5,
                record_trajectory=False
            )
            alg = SMC(config=config, log_prob=joint_log_prob)
            key, subkey = random.split(key)
            alg = alg.init(key=subkey, dist=proposal)
            key, subkey = random.split(key)
            alg, results = alg.run(key=subkey)
            zxs = results.samples.resample_and_reset_weights(subkey).xs

            zs_orig_space = posterior.z_transform(zxs[:, :self.dataset.dim_params])
            xs_orig_space = posterior.x_transform(zxs[:, self.dataset.dim_params:])
            return jnp.concatenate([zs_orig_space, xs_orig_space], axis=1)

        keys = random.split(key, num=len(iter_nos))
        return vmap(_get_joint_samples)(params, keys)

    def get_posterior_samples(self, iter_nos: List[int], key: PRNGKeyArray, num_samples: int) -> Array:
        assert self.train_results.trajectory is not None
        params = tree_map(
            lambda x: jnp.array(np.array(x)[iter_nos]),
            self.train_results.trajectory.params
        )

        def _get_posterior_samples(params, key):
            posterior = copy.copy(self.posterior)
            posterior = posterior.replace_params(params)
            posterior_log_prob = maybe_wrap(lambda x: posterior.log_prob(x))
            smc_proposal = self.posterior.prior.base_dist  # type: ignore

            config = SMCConfig(
                num_samples=1000,
                num_steps=200, ess_threshold=0.8,
                inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
                inner_kernel_steps=5,
                record_trajectory=False
            )
            alg = SMC(config=config, log_prob=posterior_log_prob)

            key, subkey = random.split(key)
            alg = alg.init(key=subkey, dist=smc_proposal)

            key, subkey = random.split(key)
            alg, results = alg.run(key=subkey)

            key, subkey = random.split(key)
            return results.samples.resample_and_reset_weights(subkey).xs

        keys = random.split(key, num=len(iter_nos))
        return vmap(_get_posterior_samples)(params, keys)


class Results(NamedTuple):
    config: Config
    posterior: EBMPosterior
    posterior_samples: Array
    single_round_results: Tuple[SingleRoundResults]
    total_time: float = 0.



class MixtureDistribution(np_distributions.Distribution):
    arg_contraints = {"distributions": None, "mixture_props": None}
    def __init__(
        self,
        distributions: Tuple[np_distributions.Distribution, ...],
        mixture_props: Array
    ):
        assert len({d.event_shape for d in distributions}) == 1
        assert len({d.batch_shape for d in distributions}) == 1


        assert len(distributions) == len(mixture_props)

        self.distributions = distributions
        self.mixture_props = mixture_props

        super(MixtureDistribution, self).__init__(
            batch_shape=distributions[0].batch_shape,
            event_shape=distributions[0].event_shape,
        )
        # all supports are supposed to be the same since the MOG proposals
        # are fitted in a z-scored space and transformed back to be contained
        # in the support of the prior (the first proposal)
        self.support = distributions[0].support

    def log_prob(self, x: Array) -> Numeric:
        log_probs = jnp.array([d.log_prob(x) for d in self.distributions])
        return logsumexp(log_probs, b=self.mixture_props)

    def add_new_component(self, dist: np_distributions.Distribution, mixture_prop: Numeric) -> Self:
        assert not isinstance(dist, EBMPosterior), "distribution must be normalized"

        new_distributions = (*self.distributions, dist)

        new_cluster_props = jnp.append(self.mixture_props, mixture_prop)
        new_cluster_props = new_cluster_props / jnp.sum(new_cluster_props)

        return MixtureDistribution(new_distributions, new_cluster_props)


LD = TypeVar("LD", LogDensity_T, DoublyIntractableLogDensity)


class MultiRoundTrainer:
    def __init__(
        self,
        trainer: Trainer,
    ):
        self.trainer = trainer

    def _build_posterior(
        self,
        prior_dist: np_distributions.Distribution,
        z_scorer: ZScorer,
        x_obs: Array,
        likelihood_factory: Optional[LikelihoodFactory] = None,
        ratio: Optional[_EBMRatio] = None,
        calibration_net: Optional[SBIEBMCalibrationNet] = None
    ) -> EBMPosterior:
        z_scored_prior_dist = np_distributions.TransformedDistribution(
            prior_dist, z_scorer.get_transform("params")
        )
        tz = z_scorer.get_transform("params").inv
        tx = z_scorer.get_transform("observations").inv

        return EBMPosterior(z_scored_prior_dist, x_obs, tz, tx, calibration_net=calibration_net, likelihood_factory=likelihood_factory, ratio=ratio)

    def _zscore_sample_posterior(
        self,
        posterior: EBMPosterior,
        sampling_config: InferenceAlgorithmFactory,
        init_dist: np_distributions.Distribution,
        key: PRNGKeyArray,
        prev_samples: Optional[Array] = None,
        init_chain_state: Optional[Tuple[Any, Any]] = None,
        round_no: int = 0,
    ) -> Tuple[Array, Array]:
        if posterior.likelihood_factory and posterior.likelihood_factory.is_doubly_intractable:
            z_scored_posterior = DoublyIntractableLogDensity(
                log_prior=maybe_wrap(lambda x: posterior._prior_zscored_space(x) + posterior.calibration_net_log_prob(x)),
                log_likelihood=maybe_wrap_log_l(posterior._log_likelihood_zscored_space),
                x_obs=posterior.x_transform.inv(posterior.x)
            )
        else:
            # the likelihood does not need a calibration correction when learning a model of the joint 
            z_scored_posterior = maybe_wrap(lambda x: posterior.log_prob_zscored_space(x))

        alg = sampling_config.build_algorithm(z_scored_posterior)

        # sample in unconstrained theta space
        if init_chain_state is not None:
            init_chain_pos, init_chain_config = init_chain_state
            print('initializing MCMC inference chain using chains from previous round')
            assert isinstance(alg, MCMCAlgorithm)
            alg = alg.init_from_particles(init_chain_pos)
            assert alg._single_chains is not None
            alg = alg.replace(_single_chains=alg._single_chains.replace(config=alg._single_chains.config.replace(kernel_factory=init_chain_config.kernel_factory)))
        else:
            key, subkey = random.split(key)
            alg = alg.init(key=subkey, dist=init_dist)

        key, subkey = random.split(key)
        alg, results = jit(alg.run)(subkey)

        posterior_samples = posterior.z_transform(results.samples.xs)

        if isinstance(results, MCMCResults):
            assert isinstance(alg, MCMCAlgorithm)
            final_chain_pos = results.info.single_chain_results.final_state.x
            assert alg._single_chains is not None
            tuned_chains_config = alg._single_chains.config
            return posterior_samples, (final_chain_pos, tuned_chains_config)
        else:
            return posterior_samples, None

    def _combine_with_past_rounds_data(
        self, dataset: SBIDataset, previous_dataset: SBIDataset,
        round_sizes: Tuple[int, ...]
    ) -> SBIDataset:

        arr_round_sizes = jnp.array(round_sizes)
        round_props = arr_round_sizes / jnp.sum(arr_round_sizes)

        assert len(round_sizes) >= 2
        if len(round_sizes) == 2:
            combined_round_proposals = (previous_dataset.prior, dataset.prior)
        else:
            previous_mixture_dist = previous_dataset.prior
            assert isinstance(previous_mixture_dist, MixtureDistribution)
            combined_round_proposals = (
                *previous_mixture_dist.distributions, dataset.prior
            )
        mixture_proposal = MixtureDistribution(combined_round_proposals, round_props)

        # standardize tree_structure by uniformizing static attributes
        std_previous_dataset = SBIDataset(
            previous_dataset.train_samples.replace(prior=mixture_proposal),
            previous_dataset.test_samples.replace(prior=mixture_proposal)
        )
        std_dataset = SBIDataset(
            dataset.train_samples.replace(prior=mixture_proposal),
            dataset.test_samples.replace(prior=mixture_proposal)
        )

        mixture_dataset: SBIDataset = tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), std_previous_dataset, std_dataset
        )

        mixture_dataset = SBIDataset(
            mixture_dataset.train_samples.replace(indices=jnp.arange(mixture_dataset.train_samples.num_samples)),
            mixture_dataset.test_samples.replace(indices=jnp.arange(mixture_dataset.test_samples.num_samples)),
        )

        print(mixture_dataset.train_samples.indices)
        return mixture_dataset

    def _prepare_init_state(
        self,
        datasets: Tuple[SBIDataset, ...],
        config: TrainingConfig,
        key: PRNGKeyArray,
        prev_state: Optional[TrainState] = None,
        calibration_net: Optional[CalibrationMLP] = None,
        params: Optional[PyTreeNode] = None,
    ) -> TrainState:
        init_state = self.trainer.initialize_state(
            datasets=datasets, config=config, key=key, use_first_iter_cfg=True,
            calibration_net=calibration_net,
            params=params
        )

        # if prev_state is not None:
        #     init_state = init_state.replace(params=prev_state.params)
        #     if len(datasets) > len(init_state.training_alg):
        #         init_state = init_state.replace(
        #             sampling_init=(
        #                 init_state.sampling_init[0],
        #                 *prev_state.sampling_init,
        #             ),
        #         )
        return init_state

    def _perturb_parameters(self, this_round_thetas: Array, task_config: TaskConfig, key: PRNGKeyArray):
        # add a bit of gaussian noise to all parameters before feeding them to the simulator.
        print('adding noise to thetas')
        key, key_noise = random.split(key)
        noise = 2. * random.normal(key_noise, shape=this_round_thetas.shape)  # type: ignore
        unnoised_thetas = this_round_thetas
        this_round_thetas = this_round_thetas + noise
        # retain the un-noised parameters if its associated noisy parameters is out of the prior bounds.
        # this is to avoid having to re-run the simulator for those parameters.
        out_of_bounds_mask = task_config.prior.log_prob(this_round_thetas) == -jnp.inf
        print("number of out of bounds noisy samples: ", jnp.sum(out_of_bounds_mask))
        this_round_thetas = jnp.where(
            out_of_bounds_mask[:, None],
            unnoised_thetas,
            this_round_thetas
        )
        return this_round_thetas

    def _transfer_to_cpu(self, tree: PyTreeNode) -> None:
        all_leaves = tree_leaves(
            tree,
            is_leaf=lambda x: isinstance(x, np_distributions.TransformedDistribution)
        )
        prev_arrays = [leaf for leaf in all_leaves if isinstance(leaf, jnp.ndarray)]
        jax.device_put(prev_arrays, device=jax.devices("cpu")[0])

    def _adjust_traning_config(self, config: TrainingConfig) -> TrainingConfig:
        # config = config.replace(
        #     optimizer=config.optimizer._replace(
        #         learning_rate=config.optimizer.learning_rate / 2
        #     )
        # )
        # config = config.replace(
        #     sampling_cfg=config.sampling_cfg.replace(
        #         num_steps=config.sampling_cfg.num_steps * 2)
        # )
        # return config
        return config


    def train_calibration_net(self, params: Array, y: Array) -> SBIEBMCalibrationNet:


        from sklearn.model_selection import train_test_split
        theta_train, theta_test, y_train, y_test = train_test_split(
            params, y, random_state=43, stratify=y, train_size=0.8
        )
        assert not isinstance(theta_train, list)

        rng = jax.random.PRNGKey(0)

        rng, init_rng = jax.random.split(rng)
        from sbi_ebm.calibration.calibration import create_train_state, train_epoch, apply_model, logging
        state = create_train_state(init_rng, theta_train)

        batch_size = 10000
        batch_size = min(batch_size, theta_train.shape[0])

        max_iter = 200

        class_weights = jnp.array(
            [1 / (y_train == 0).sum(), 1 / (y_train == 1).sum()]  # type: ignore
        )
        for epoch in range(1, max_iter):
            rng, input_rng = jax.random.split(rng)
            state, train_loss, (train_accuracy, train_a0, train_a1) = train_epoch(
                state, (theta_train, y_train), batch_size, input_rng, class_weights
            )
            # print(train_accuracy)
            _, test_loss, (test_accuracy, test_a0, test_a1) = apply_model(
                state, theta_test, y_test, class_weights
            )

            if (epoch % max(max_iter // 20, 1)) == 0:
                # logging.info(
                #     "epoch:% 3d, train_loss: %.6f, train_accuracy: %.4f, test_loss: %.6f, test_accuracy: %.4f"
                #     % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
                # )
                logging.info(
                    "epoch:% 3d, train_a0: %.4f, train_a1: %.4f, test_a0: %.4f, test_a1: %.4f, test_accuracy: %.4f"
                    % (epoch, train_a0 * 100, train_a1 * 100, test_a0 * 100, test_a1 * 100, test_accuracy * 100)
                )

        return SBIEBMCalibrationNet(state.params)

    def _can_reuse_data_from_unnormalized_proposal(self, config: Config) -> bool:
        return config.training.ebm_model_type in ("ratio", "likelihood")

    def train_sbi_ebm(self, config: Config, key: PRNGKeyArray) -> Results:
        # TODO: extract the single round training logic into a separate function
        t0_sbi_ebm = time.time()
        num_rounds = len(config.num_samples)
        prev_state = None
        prev_z_scorer = None
        prev_inference_state = None

        training_config, task_config = config.training, config.task

        if config.checkpointing.should_start_from_checkpoint:
            if config.checkpointing.single_round_results is not None:
                single_round_results = config.checkpointing.single_round_results
            else:
                init_checkpoint_path = config.checkpointing.init_checkpoint_path
                print("loading data from checkpoint file:", init_checkpoint_path)
                import pickle
                with open(init_checkpoint_path, "rb") as f:
                    single_round_results: List[SingleRoundResults] = pickle.load(f)


            this_round_results = single_round_results[config.checkpointing.init_checkpoint_round]
            proposal_dist = this_round_results.posterior

            maybe_previous_complete_dataset = this_round_results.complete_dataset
            if maybe_previous_complete_dataset is None:
                # XXX: do it better than this
                previous_complete_dataset = this_round_results.dataset.train_samples
            else:
                previous_complete_dataset = maybe_previous_complete_dataset


            this_round_thetas = this_round_results.posterior_samples

            current_posterior = this_round_results.posterior
            current_posterior_samples = this_round_results.posterior_samples

            if config.checkpointing.init_checkpoint_round == -1:
                init_round_no = len(single_round_results)
            else:
                assert config.checkpointing.init_checkpoint_round >= 0
                init_round_no = config.checkpointing.init_checkpoint_round + 1
            print("resuming training at round :", init_round_no)
            if len(single_round_results) > init_round_no:
                # XXX only if the raw posterior was used as the proposal
                precomputed_thetas = jnp.concatenate([
                    single_round_results[init_round_no].dataset.train_samples.params,
                    single_round_results[init_round_no].dataset.test_samples.params,
                ], axis=0)
                precomputed_obsevations = jnp.concatenate([
                    single_round_results[init_round_no].dataset.train_samples.observations,
                    single_round_results[init_round_no].dataset.test_samples.observations,
                ], axis=0)
            else:
                precomputed_obsevations = None
                precomputed_thetas = None


            single_round_results = single_round_results[:init_round_no]


            if len(this_round_thetas) > config.num_samples[init_round_no]:
                print('shrinking last number of proposal samples to: ', config.num_samples[init_round_no])
                this_round_thetas = this_round_thetas[:config.num_samples[init_round_no]]
                current_posterior_samples = current_posterior_samples[:config.num_samples[init_round_no]]
            elif len(this_round_thetas) < config.num_samples[init_round_no]:
                raise ValueError('not enough samples in the checkpointed proposal')


            all_datasets: Tuple[SBIDataset, ...] = tuple([r.dataset for r in single_round_results])

        else:
            previous_complete_dataset = None
            precomputed_obsevations = None
            precomputed_thetas = None
            proposal_dist = config.proposal.init_proposal

            # only smooth the initial proposal when benchmarking the use of the reference
            # posterior as a proposal
            if isinstance(proposal_dist, ReferencePosterior) and config.proposal.tempering_enabled:
                key, key_tempered_approx = random.split(key)
                assert config.proposal.mog_config is not None
                proposal_dist = KDEApproxDist(
                    proposal_dist, smc_proposal=task_config.prior,
                    key=key_tempered_approx, mog_config=config.proposal.mog_config,
                    fit_in_z_score_space=True, t_prior=config.proposal.t_prior
                )

            key, key_proposal = random.split(key)
            this_round_thetas = proposal_dist.sample(key_proposal, (config.num_samples[0],))
            init_round_no = 0
            single_round_results: List[SingleRoundResults] = []
            this_round_results = None
            current_posterior, current_posterior_samples = None, None

            all_datasets: Tuple[SBIDataset, ...] = tuple()
        has_nan: bool = False

        init_training_init_dist = training_config.sampling_init_dist
        init_inference_init_dist = config.inference.sampling_init_dist

        complete_dataset = None
        n_sigma = config.n_sigma

        for round_no in range(init_round_no, num_rounds):
            t0_simulation = time.time()
            if round_no == 0 and config.task.task_name == "pyloric":
                print("loading cached training samples for round 0")
                import pickle
                pyloric_data_filename = os.path.join(os.path.dirname(__file__), 'theta_and_x_pyloric.pkl')
                with open(pyloric_data_filename, "rb") as f:
                    this_round_thetas_torch, this_round_xs_torch = pickle.load(f)
                    this_round_thetas = jnp.array(this_round_thetas_torch.clone().detach().numpy())
                    this_round_xs = jnp.array(this_round_xs_torch.clone().detach().numpy())

                    this_round_thetas = this_round_thetas[:config.num_samples[0]]  # type: ignore
                    this_round_xs = this_round_xs[:config.num_samples[0]] # type: ignore

                    print("loaded {} samples".format(len(this_round_thetas)))
                    print("loaded {} samples".format(len(this_round_xs)))

            elif round_no == init_round_no and config.checkpointing.should_start_from_checkpoint:
                print("trying to get possibly already computed simulations from next round")
                if precomputed_obsevations is not None and precomputed_thetas is not None:
                    this_round_thetas = precomputed_thetas
                    this_round_xs = precomputed_obsevations
                    print("used a precomputed dataset")
                else:
                    this_round_xs = task_config.simulator(this_round_thetas)

            else:
                # key, key_proposal = random.split(key)
                # this_round_thetas = config.proposal.init_proposal.sample(key_proposal, (config.num_samples[round_no],))
                this_round_xs = task_config.simulator(this_round_thetas)


            _this_round_thetas_all = this_round_thetas
            _this_round_xs_all = this_round_xs
            this_round_complete_dataset = SBIParticles.create(_this_round_thetas_all, _this_round_xs_all, prior=None)  # type: ignore

            if task_config.use_calibration_kernel:
                is_valid_sim = jnp.sum(jnp.isnan(this_round_xs), axis=1) == 0
                print(f"num valid simulations: {jnp.sum(is_valid_sim)}", flush=True)


                this_round_xs = this_round_xs[is_valid_sim]  # type: ignore
                this_round_thetas = this_round_thetas[is_valid_sim]  # type: ignore

            simulation_time = time.time() - t0_simulation
            import datetime
            print("generating data took time: ", str(datetime.timedelta(seconds=int(simulation_time))))

            assert proposal_dist is not None  # type checker...
            print(f"creating dataset with {len(this_round_xs)} samples")  # type: ignore
            dataset = SBIDataset.create(
                params=this_round_thetas,
                observations=this_round_xs,
                frac_test_samples=config.frac_test_samples,
                prior=proposal_dist,
            )
            all_datasets = (*all_datasets, dataset)

            z_scored_dataset, z_scorer = _z_score_proposal_and_data(
                dataset,
                normalize=config.preprocessing.normalize,
                biject_to_unconstrained_space=config.preprocessing.biject_to_unconstrained_space
            )


            if (config.use_data_from_past_rounds and ((not config.discard_prior_samples and round_no >= 1) or (config.discard_prior_samples and round_no >= 2))):
                first_round_idx = 1 if config.discard_prior_samples else 0
                if config.proposal.tempering_enabled:
                    print("combining normalized data: combining all datasets into a normalized mixture dataset.")

                    # tempering uses a MOG proposal, making them normalized, allowing to
                    # compare samples across rounds
                    # XXX: add a round_no array attribute in SBIDataset
                    assert this_round_results is not None
                    dataset = self._combine_with_past_rounds_data(
                        dataset, this_round_results.dataset,
                        config.num_samples[first_round_idx:round_no+1]
                    )
                    z_scored_dataset, z_scorer = _z_score_proposal_and_data(
                        dataset,
                        normalize=config.preprocessing.normalize,
                        biject_to_unconstrained_space=config.preprocessing.biject_to_unconstrained_space
                    )

                    # use last round of theta samples to z-score the entire aggregated dataset
                    # z_scored_dataset = SBIDataset(
                    #     train_samples=z_scorer.transform(dataset.train_samples),
                    #     test_samples=z_scorer.transform(dataset.test_samples),
                    # )
                    self._transfer_to_cpu(this_round_results)

                    # all_z_scored_datasets = (z_scored_dataset, *all_z_scored_datasets)
                    all_z_scored_datasets = (z_scored_dataset,)

                    _outlier_mask_xs = None
                    outlier_dataset = None

                elif self._can_reuse_data_from_unnormalized_proposal(config):
                    print('combining unnormalized data with unle-likelihood: merging all datasets')
                    # case of an unnormalized likelihood EBM which does not rely on prior probabilities at all: safe to concatenate all datasets
                    combined_dataset = SBIDataset(
                        train_samples=SBIParticles.create(
                            params=jnp.concatenate([d.train_samples.params for d in all_datasets[first_round_idx:]]),
                            observations=jnp.concatenate([d.train_samples.observations for d in all_datasets[first_round_idx:]]),
                            prior=proposal_dist,  # smoke attribute
                            log_ws=jnp.concatenate([d.train_samples.log_ws for d in all_datasets[first_round_idx:]]),
                        ),
                        test_samples=SBIParticles.create(
                            params=jnp.concatenate([d.test_samples.params for d in all_datasets[first_round_idx:]]),
                            observations=jnp.concatenate([d.test_samples.observations for d in all_datasets[first_round_idx:]]),
                            prior=proposal_dist,  # smoke attribute
                            log_ws=jnp.concatenate([d.test_samples.log_ws for d in all_datasets[first_round_idx:]]),
                        )
                    )
                    # z_scored_dataset, z_scorer = _z_score_proposal_and_data(
                    #     combined_dataset,
                    #     normalize=config.preprocessing.normalize,
                    #     biject_to_unconstrained_space=config.preprocessing.biject_to_unconstrained_space
                    # )
                    # get number of samples in combined dataset over 4 sigmas of the last dataset
                    # (4 is a magic number, but it's a good one)
                    # use last round of theta samples to z-score the entire aggregated dataset
                    train_and_test_samples = jnp.concatenate(
                        [combined_dataset.train_samples.xs, combined_dataset.test_samples.xs], axis=0
                    )

                    # _this_round_mean = jnp.concatenate(
                    #     [jnp.mean(this_round_thetas, axis=0), jnp.mean(this_round_xs, axis=0)]
                    # )
                    # _this_round_std = jnp.concatenate(
                    #     [jnp.std(this_round_thetas, axis=0), jnp.std(this_round_xs, axis=0)]
                    # ) + 1e-8  # type: ignore

                    # _outlier_mask = jnp.any(
                    #     jnp.abs(train_and_test_samples - _this_round_mean) > 4. * _this_round_std,  # type: ignore
                    #     axis=1
                    # )

                    # Filter out outlier observations: requires a likelihood calibration step to account for the biase induced on
                    # likelihood estimation

                    _this_round_mean_xs = jnp.mean(this_round_xs, axis=0)
                    _this_round_std_xs = jnp.std(this_round_xs, axis=0) + 1e-8  # type: ignore
                    _outlier_mask_xs = jnp.sqrt(jnp.sum(jnp.square(train_and_test_samples[:, config.task.dim_parameters:] - _this_round_mean_xs), axis=1)) > n_sigma * jnp.sqrt(jnp.sum(jnp.square(_this_round_std_xs)))
                    # _outlier_mask_xs = jnp.any(
                    #     jnp.abs(train_and_test_samples[:, config.task.dim_parameters:] - _this_round_mean_xs) > n_sigma * _this_round_std_xs,  # type: ignore
                    #     axis=1
                    # )
                    num_outlier_obs = jnp.sum(_outlier_mask_xs)
                    print(f"number of samples in combined dataset with observation over {n_sigma} sigmas: {num_outlier_obs}")

                    if num_outlier_obs > 0:
                        outlier_dataset = (_this_round_mean_xs, _this_round_std_xs)
                    else:
                        outlier_dataset = None

                    non_outlier_samples = train_and_test_samples[~_outlier_mask_xs]  # type: ignore
                        
                    # Now, filter outlier parameters: this does not bias the likelihood.
                    _this_round_mean = jnp.mean(this_round_thetas, axis=0)
                    _this_round_std = jnp.std(this_round_thetas, axis=0) + 1e-8  # type: ignore

                    _outlier_mask_thetas = jnp.any(
                        jnp.abs(non_outlier_samples[:, :config.task.dim_parameters] - _this_round_mean) > n_sigma * _this_round_std,  # type: ignore
                        axis=1
                    )
                    num_outlier_thetas = jnp.sum(_outlier_mask_thetas)
                    print(f"number of samples in combined dataset over {n_sigma} sigmas: {num_outlier_thetas}")

                    non_outlier_samples = non_outlier_samples[~_outlier_mask_thetas]  # type: ignore

                    key, subkey = random.split(key)
                    non_outlier_combined_dataset = SBIDataset.create(
                        params=non_outlier_samples[:, :config.task.dim_parameters],
                        observations=non_outlier_samples[:, config.task.dim_parameters:],
                        prior=proposal_dist,  # smoke attribute
                        frac_test_samples=0.05,
                        key=subkey,
                    )
                    z_scored_dataset = SBIDataset(
                        train_samples=z_scorer.transform(non_outlier_combined_dataset.train_samples),
                        test_samples=z_scorer.transform(non_outlier_combined_dataset.test_samples),
                    )
                    all_z_scored_datasets = (z_scored_dataset,)
                else:
                    print('combining unnormalized data with unle-tilted: reusing past datasets individually')
                    all_z_scored_datasets = (
                        *[SBIDataset(train_samples=z_scorer.transform(d.train_samples), test_samples=z_scorer.transform(d.test_samples)) for d in all_datasets[first_round_idx:-1]],
                        z_scored_dataset,
                    )
                    _outlier_mask_xs = None
                    outlier_dataset = None
            else:
                if round_no > 0:
                    print('not reusing data from previous rounds.')
                all_z_scored_datasets = (z_scored_dataset,)
                _outlier_mask_xs = None
                outlier_dataset = None

    
            if isinstance(init_training_init_dist, BlockDistribution):
                print("z_scoring theta-part of sampling init dist")
                training_init_dist_z_scored_theta = copy.deepcopy(init_training_init_dist)

                training_init_dist_z_scored_theta.distributions[0] = np_distributions.TransformedDistribution(
                    training_init_dist_z_scored_theta.distributions[0], z_scorer.get_transform("params")
                )
                training_config = training_config.replace(
                    sampling_init_dist=training_init_dist_z_scored_theta,
                    likelihood_estimation_config=training_config.likelihood_estimation_config.replace(
                        init_dist=training_init_dist_z_scored_theta
                    )
                )
            # if isinstance(init_inference_init_dist, BlockDistribution)
            # XXX: it only makes sense to z-score the inference_init_dist if we use the prior.
            if config.inference.should_z_score:
                print("z_scoring theta-part of inference init dist")
                inference_init_dist_z_scored_theta = copy.deepcopy(init_inference_init_dist)
                inference_init_dist_z_scored_theta = np_distributions.TransformedDistribution(
                    inference_init_dist_z_scored_theta, z_scorer.get_transform("params")
                )

                config = config.replace(
                    inference=config.inference.replace(sampling_init_dist=inference_init_dist_z_scored_theta)
                )

                print(f"{config.inference.sampling_init_dist=}")

            # log_probs_test = vmap(z_scored_dataset.train_samples.prior.log_prob)(
            #     z_scored_dataset.train_samples.params
            # )
            # assert jnp.all(jnp.isfinite(log_probs_test))

            if isinstance(config.training.sampling_cfg, SMCConfig):
                if training_config.sampling_init_dist == "data":
                    key, subkey = random.split(key)
                    _mog_config = MOGTrainingConfig(
                        num_clusters=300, min_std=0.25, max_iter=100, num_inits=2
                    )
                    _mog = fit_mog(
                        dataset.train_samples.xs, _mog_config, key=subkey
                    ).to_dist()
                    training_config = training_config.replace(sampling_init_dist=_mog)


            # train EBM
            key, key_training = random.split(key)
            training_converged = False
            max_retries = 1
            retry_no = 0

            if round_no == init_round_no:
                if previous_complete_dataset is not None:
                    complete_dataset = previous_complete_dataset.replace(prior=None)
                else:
                    complete_dataset = None

            if complete_dataset is None:
                assert round_no == init_round_no
                complete_dataset = this_round_complete_dataset
            else:
                complete_dataset = cast(SBIParticles, tree_map(
                    lambda *xs: jnp.concatenate(xs, axis=0), complete_dataset, this_round_complete_dataset
                ))
            assert complete_dataset is not None

            if task_config.use_calibration_kernel or outlier_dataset is not None:

                if outlier_dataset is not None:
                    print('some outlier samples were detected and discarded from the dataset, fitting a calibration kernel to debias the likelihood.')
                    this_round_mean_xs, this_round_std_xs = outlier_dataset
                    non_outlier_mask = jnp.sqrt(jnp.sum(jnp.square(complete_dataset.observations - this_round_mean_xs), axis=1)) < n_sigma * jnp.sqrt(jnp.sum(jnp.square(this_round_std_xs)))
                else:
                    non_outlier_mask = jnp.ones(complete_dataset.observations.shape[0], dtype=bool)

                if jnp.any(jnp.isnan(complete_dataset.observations)):
                    print('This model contains some invalid data that need to be discarded: fitting a calibration kernel to debias the likelihood.')
                    non_nan_mask = (jnp.sum(jnp.isnan(complete_dataset.observations), axis=1) == 0)
                else:
                    non_nan_mask = jnp.ones(complete_dataset.observations.shape[0], dtype=bool)

                valid_mask = (non_outlier_mask & non_nan_mask).astype(jnp.float32)

                all_thetas_zscored = z_scorer.get_transform(who='params').__call__(complete_dataset.params)
                print(all_thetas_zscored)
                print(f"fitting a calibration kernel with {len(complete_dataset.particles)} samples and {jnp.sum(valid_mask)} valid samples")

                non_outlier_theta_mask = jnp.sqrt(jnp.sum(jnp.square(all_thetas_zscored), axis=1)) < jnp.sqrt(jnp.sum(jnp.square(jnp.std(all_thetas_zscored, axis=0)))) * n_sigma
                all_thetas_zscored = all_thetas_zscored[non_outlier_theta_mask]
                valid_mask = valid_mask[non_outlier_theta_mask]

                print(f"using {jnp.sum(non_outlier_theta_mask)} non-outlier samples to fit calibration network")

                if sum(valid_mask) >= len(all_thetas_zscored) - 1:
                    # avoid sklern issues due 1-sample class
                    print('only one invalid sample found, skipping calibration step')
                    calibration_net = None
                else:
                    calibration_net = self.train_calibration_net(all_thetas_zscored, valid_mask)

                key, key_init = random.split(key)
                init_state = self._prepare_init_state(
                    all_z_scored_datasets, training_config, key_init, prev_state=None,
                    calibration_net=None  # type: ignore
                )
                # init_state = init_state.replace(calibration_net=calibration_net)
                _cn  = calibration_net
                calibration_net = None

            else:
                # prepare initial traning step (with potential warm start)
                key, key_init = random.split(key)
                if round_no == 0:
                    init_state = self._prepare_init_state(
                        all_z_scored_datasets, training_config, key_init, prev_state=None
                    )
                else:
                    init_state = self._prepare_init_state(
                        # all_z_scored_datasets, training_config, key_init, prev_state=training_results.init_state
                        all_z_scored_datasets, training_config, key_init, prev_state=None
                    )
                _cn = None
                calibration_net = None


            print('training likelihood model...')
            # warm start: need to perform params surgery to account for new z-scoring

            if round_no > 0:
                print("warm starting optimization using final params from last round")
                assert prev_state is not None
                assert prev_z_scorer is not None
                init_params = rescale_params(prev_state.params, prev_z_scorer, z_scorer)
                init_state = init_state.replace(params=init_params)
                # training_config = training_config.replace(max_iter=5)

            training_results = self.trainer.train_ebm_likelihood_model(
                datasets=all_z_scored_datasets,
                config=training_config,
                key=key_training,
                init_state=init_state,
            )


            while not training_converged and retry_no <= max_retries:
                training_converged = True or not training_results.final_state.opt_is_diverging
                if training_converged:
                    break
                else:
                    print(
                        "optimization was unstable, increasing number of sampler steps"
                    )
                    training_config = self._adjust_traning_config(training_config)
                    training_results = self.trainer.train_ebm_likelihood_model(
                        datasets=all_z_scored_datasets,
                        config=training_config,
                        key=key_training,
                        init_state=init_state,
                    )
                    retry_no += 1

            print(f"Best state found after {training_results.best_state.step} iterations")

            # build next round proposal and samples
            key, key_sampling = random.split(key)
            if round_no + 1 == num_rounds:
                this_round_num_samples = config.inference.num_samples
            else:
                this_round_num_samples = config.num_samples[round_no + 1]


            # calibration net should be none as it was used to learn an unbiased likelihood during training.
            print(f"calibration net: {calibration_net}")
            print(f"calibration net (init): {init_state.calibration_net}")
            current_posterior = self._build_posterior(
                task_config.prior,
                z_scorer,
                task_config.x_obs,
                likelihood_factory=training_results.likelihood_factory,
                ratio=training_results.ratio,
                calibration_net=_cn if config.training.ebm_model_type in ("likelihood",) else None # type: ignore
            )

            if this_round_num_samples == 0:
                current_posterior_samples = None
                inference_time = 0.0
                inference_state = None
            else:
                print('sampling from posterior...')
                t0 = time.time()
                if prev_inference_state is not None:
                    final_chain_pos, tuned_chain_config = prev_inference_state
                    assert prev_z_scorer is not None
                    final_chain_pos_rescaled = vmap(rescale_zscored_sample, in_axes=(0, None, None))(final_chain_pos, prev_z_scorer, z_scorer)
                    from sbi_ebm.samplers.inference_algorithms.mcmc.base import _MCMCChainConfig
                    assert isinstance(tuned_chain_config, _MCMCChainConfig)

                    zscoring_ratio = get_zscoring_ratio(prev_z_scorer, z_scorer)
                    print('zscoring ratio', zscoring_ratio)

                    if isinstance(tuned_chain_config.kernel_factory.config, SAVMConfig):
                        tuned_chain_config = tuned_chain_config.replace(kernel_factory=tuned_chain_config.kernel_factory.replace(
                            config=tuned_chain_config.kernel_factory.config.replace(
                                base_var_kernel_factory=tuned_chain_config.kernel_factory.config.base_var_kernel_factory.replace(
                                config=tuned_chain_config.kernel_factory.config.base_var_kernel_factory.config.replace(
                                    step_size=tuned_chain_config.kernel_factory.config.base_var_kernel_factory.config.step_size * zscoring_ratio)
                        ))))
                    else:
                        tuned_chain_config = tuned_chain_config.replace(kernel_factory=tuned_chain_config.kernel_factory.replace(
                            config=tuned_chain_config.kernel_factory.config.replace(step_size=tuned_chain_config.kernel_factory.config.step_size * zscoring_ratio)
                        ))

                    prev_inference_state = (final_chain_pos_rescaled, tuned_chain_config)


                current_posterior_samples, inference_state = self._zscore_sample_posterior(
                    current_posterior,
                    #config.inference.sampling_config.replace(config=config.inference.sampling_config.config.replace(num_samples=this_round_num_samples)) if round_no == 0 else config.inference.sampling_config.replace(config=config.inference.sampling_config.config.replace(num_samples=this_round_num_samples, num_warmup_steps=10)),
                    config.inference.sampling_config.replace(config=config.inference.sampling_config.config.replace(num_samples=this_round_num_samples)),
                    config.inference.sampling_init_dist,
                    key_sampling,
                    init_chain_state=prev_inference_state,
                    round_no=round_no,
                )
                inference_time = time.time() - t0
                print("inference took time: ", str(datetime.timedelta(seconds=int(inference_time))))

            if round_no < num_rounds - 1:
                if config.proposal.tempering_enabled:
                    print("computing a normalized, defensive proposal using a MoG model.")
                    assert config.proposal.mog_config is not None
                    key, key_tempered_approx = random.split(key)
                    proposal_dist = KDEApproxDist(
                        current_posterior, smc_proposal=task_config.prior,
                        key=key_tempered_approx, mog_config=config.proposal.mog_config,
                        # key=key_tempered_approx, mog_config=config.proposal.mog_config.replace(cov_reg_param=jnp.var(current_posterior_samples, axis=0)),
                        fit_in_z_score_space=True,
                        train_samples=current_posterior_samples,
                        t_prior=config.proposal.t_prior
                    )
                    key, key_proposal = random.split(key)
                    this_round_thetas = proposal_dist.sample(
                        key_proposal, (this_round_num_samples,)
                    )

                    _lps = vmap(proposal_dist.log_prob)(this_round_thetas)
                    assert jnp.all(jnp.isfinite(_lps))
                    assert jnp.all(jnp.isfinite(this_round_thetas))

                else:
                    proposal_dist = current_posterior
                    this_round_thetas = current_posterior_samples

            print(len(this_round_thetas))  # type: ignore

            this_round_results = SingleRoundResults(
                config=config,
                dataset=dataset,
                z_scorer=z_scorer,
                posterior=current_posterior,
                posterior_samples=current_posterior_samples,
                train_results=training_results,
                x_obs=task_config.x_obs,
                complete_dataset=complete_dataset,
                simulation_time=simulation_time,
                inference_time=inference_time,
            )

            single_round_results.append(this_round_results)

            if config.checkpointing.should_checkpoint:
                from pathlib import Path
                checkpoint_path = Path(config.checkpointing.checkpoint_path)

                print("saving results")
                print(f"recording results in {checkpoint_path}")
                import cloudpickle
                with open(checkpoint_path.with_suffix('.pkl'+ str(round_no)), "wb") as f:
                    cloudpickle.dump(single_round_results, f)

                with open(checkpoint_path, "wb") as f:
                    cloudpickle.dump(single_round_results, f)


            if this_round_results.train_results.best_state.has_nan:
                has_nan = True
                break

            prev_state = training_results.best_state
            prev_z_scorer = z_scorer
            prev_inference_state = inference_state


        assert current_posterior is not None
        # assert current_posterior_samples is not None
        results = Results(
            config=config,
            posterior=current_posterior,
            posterior_samples=current_posterior_samples,
            single_round_results=tuple(single_round_results),
            total_time=time.time() - t0_sbi_ebm
        )
        print(f"sbi_ebm completed in {results.total_time} seconds")
        # if has_nan:
        #     print("stopping procedure early, found nans")
        #     from uuid import uuid4
        #     import cloudpickle
        #     filename = f"{uuid4().hex}.pkl"
        #     print(f"recording results in {filename}")
        #     with open(filename, "wb") as f:
        #         cloudpickle.dump(results, f)


        return results
