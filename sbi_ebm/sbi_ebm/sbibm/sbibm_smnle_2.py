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



class SMNLETrainEvalResults(NamedTuple):
    train_results: SMNLETrainResults
    eval_results: Optional[Any]


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


from sbi_ebm.sbi_ebm_smnle import SMNLEConfig 


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

    task_config = TaskConfig(
        simulator, convert_dist(prior, implementation="numpyro"),
        jnp.array(observation), task.name, num_observation if num_observation is not None else 0, use_calibration_kernel=False
    )

    config = SMNLEConfig(
        task=task_config,
        num_observation=num_observation,
        num_rounds=num_rounds,
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
        use_jax_mcmc=use_jax_mcmc,
        use_data_from_past_rounds=use_data_from_past_rounds,
        evaluate_posterior=evaluate_posterior,
    )


    from sbi_ebm.sbi_ebm_smnle import MultiRoundTrainer
    m = MultiRoundTrainer()
    key, subkey = random.split(key)
    train_results = m.train_sbi_ebm(config, key=subkey)
    return train_results
