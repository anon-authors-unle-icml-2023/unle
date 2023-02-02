import socket
from sbi_ebm.likelihood_trainer import LikelihoodTrainer
from sbi_ebm.samplers.inference_algorithms.base import InferenceAlgorithmFactory

from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import AdaptiveSMCConfig, AdaptiveSMCFactory, SMCConfig, SMCFactory
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm, MCMCAlgorithmFactory, MCMCConfig
from sbi_ebm.samplers.kernels.adaptive_mala import AdaptiveMALAConfig, AdaptiveMALAInfo, AdaptiveMALAKernelFactory, AdaptiveMALAState

from sbi_ebm.samplers.kernels.discrete_gibbs import DiscreteGibbsConfig, DiscreteGibbsInfo, DiscreteGibbsKernelFactory, DiscreteGibbsState
from sbi_ebm.samplers.kernels.hmc import HMCConfig, HMCKernelFactory
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAInfo, MALAKernelFactory, MALAState
from sbi_ebm.samplers.kernels.metropolis_within_gibbs import MWGConfig, MWGInfo, MWGKernelFactory, MWGState
from sbi_ebm.samplers.kernels.rwmh import RWConfig, RWKernelFactory
from sbi_ebm.samplers.kernels.savm import SAVMConfig, SAVMKernelFactory
from sbi_ebm.samplers.kernels.ula import ULAConfig, ULAInfo, ULAKernelFactory, ULAState
from sbi_ebm.samplers.kernels.numpyro_nuts import NUTSConfig, NUTSInfo, NUTSKernelFactory, NUTSState

hostname = socket.gethostname()
from sbi_ebm.dtypes import should_use_float64

if should_use_float64():
    from jax import config
    config.update('jax_enable_x64', True)


import random as python_random
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union, Any
from typing_extensions import TypeAlias

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as np_distributions
import torch
from jax import random
from sbibm.tasks.task import Task
from torch.types import Number
from typing_extensions import TypeGuard
from sbi_ebm.distributions import BlockDistribution

from sbi_ebm.likelihood_ebm import (EBMLikelihoodConfig, LikelihoodEstimationConfig, OptimizerConfig, Trainer, TrainingConfig)
from sbi_ebm.mog import MOGTrainingConfig
from sbi_ebm.pytypes import Array
from sbi_ebm.sbi_ebm import (CheckpointConfig, Config, InferenceConfig, KDEApproxDist, MultiRoundTrainer, PreProcessingConfig,
                             ProposalConfig, Results, SingleRoundResults, TaskConfig)
from sbi_ebm.sbibm.tasks import JaxTask, get_reference_posterior, get_task


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    np.random.seed(seed)
    python_random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


_r_joint_gibbs = Tuple[
    Tuple[MCMCAlgorithmFactory[DiscreteGibbsConfig, DiscreteGibbsState, DiscreteGibbsInfo], MCMCAlgorithmFactory[DiscreteGibbsConfig, DiscreteGibbsState, DiscreteGibbsInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]

_r_ula = Tuple[
    Tuple[MCMCAlgorithmFactory[ULAConfig, ULAState, ULAInfo], MCMCAlgorithmFactory[ULAConfig, ULAState, ULAInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]

_r_mala = Tuple[
    Tuple[MCMCAlgorithmFactory[MALAConfig, MALAState, MALAInfo], MCMCAlgorithmFactory[MALAConfig, MALAState, MALAInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]

_r_mwg = Tuple[
    Tuple[MCMCAlgorithmFactory[MWGConfig, MWGState, MWGInfo], MCMCAlgorithmFactory[MWGConfig, MWGState, MWGInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]


_r_nuts = Tuple[
    Tuple[MCMCAlgorithmFactory[NUTSConfig, NUTSState, NUTSInfo], MCMCAlgorithmFactory[NUTSConfig, NUTSState, NUTSInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]

_r_adaptive_mala = Tuple[
    Tuple[MCMCAlgorithmFactory[AdaptiveMALAConfig, AdaptiveMALAState, AdaptiveMALAInfo], MCMCAlgorithmFactory[AdaptiveMALAConfig, AdaptiveMALAState, AdaptiveMALAInfo]],
    Union[np_distributions.Distribution, Literal["data"]]
]

_r_smc = Tuple[Tuple[SMCFactory, SMCFactory], Union[np_distributions.Distribution, Literal["data"]]]

_sampler_init_dist_T: TypeAlias = Union[Literal["data"], np_distributions.Distribution]


def _make_train_sampler_init_dist(
    task_config: TaskConfig, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str
) -> _sampler_init_dist_T:
    if ebm_model_type == "likelihood":
        total_dim = task_config.dim_data
    else:
        total_dim = task_config.dim_parameters + task_config.dim_data

    if proposal == "noise":
        init_dist = np_distributions.MultivariateNormal(
            jnp.zeros((total_dim,)), jnp.eye(total_dim)
        )
    elif proposal == "prior+noise":
        assert ebm_model_type != "likelihood"
        x_dist = np_distributions.MultivariateNormal(
            jnp.zeros((task_config.dim_data,)), jnp.eye(task_config.dim_data),
        )
        theta_dist = task_config.prior
        init_dist = BlockDistribution(distributions=[theta_dist, x_dist])
    elif proposal == "data":
        init_dist = "data"
    else:
        raise ValueError(f"Unknown proposal {proposal}")

    return init_dist


def _make_inference_sampler_init_dist(task_config: TaskConfig, proposal: Literal["prior", "noise"]) -> np_distributions.Distribution:
    if proposal == "noise":
        posterior_sampling_init_dist = np_distributions.MultivariateNormal(
            jnp.zeros((task_config.dim_parameters,)), jnp.eye(task_config.dim_parameters)
        )
    elif proposal == "prior":
        posterior_sampling_init_dist = task_config.prior
    else:
        raise ValueError(f"Unknown proposal {proposal}")
    return posterior_sampling_init_dist


def _get_default_train_gibbs_sampling_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str
) -> _r_joint_gibbs:
    config=MCMCConfig(
        kernel_factory=DiscreteGibbsKernelFactory(config=DiscreteGibbsConfig()),
        num_samples=1000,
        num_chains=1000,
        thinning_factor=10,
        num_warmup_steps=num_mala_steps,
        adapt_mass_matrix=False,
        adapt_step_size=False
    )
    config_first_iter = config.replace(num_warmup_steps=500)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist

def _get_default_train_ula_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str
) -> _r_ula:
    config=MCMCConfig(
        kernel_factory=ULAKernelFactory(config=ULAConfig(0.001)),
        num_samples=1000,
        num_chains=1000,
        thinning_factor=10,
        num_warmup_steps=num_mala_steps,
        adapt_mass_matrix=False,
        adapt_step_size=False
    )
    config_first_iter = config.replace(num_warmup_steps=10)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist


def _get_default_train_mala_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str,
    num_frozen_steps: int = 50
) -> _r_mala:
    if ebm_model_type == "likelihood":
        imm = jnp.ones((task_config.dim_data,))
    else:
        imm = jnp.ones((task_config.dim_data + task_config.dim_parameters,))

    config=MCMCConfig(
        kernel_factory=MALAKernelFactory(config=MALAConfig(0.1, None)),
        # kernel_factory=HMCKernelFactory(HMCConfig(step_size=0.1, inverse_mass_matrix=imm, num_integration_steps=1)),
        # kernel_factory=NUTSKernelFactory(NUTSConfig(step_size=0.1, inverse_mass_matrix=imm)),
        num_samples=1000,
        num_chains=1000,
        # thinning_factor=100,
        # thinning_factor=50,  # LAST ROUNDS
        thinning_factor=num_frozen_steps,
        # thinning_factor=1,
        num_warmup_steps=num_mala_steps,
        adapt_step_size=True,
        init_using_log_l_mode=False
        # target_accept_rate=0.2,
    )
    config_first_iter = config.replace(num_warmup_steps=100)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist


def _get_default_train_mwg_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str,
    num_particles: int
) -> _r_mwg:
    if ebm_model_type == "likelihood":
        imm = jnp.ones((task_config.dim_data,))
    else:
        imm = jnp.ones((task_config.dim_data + task_config.dim_parameters,))

    config=MCMCConfig(
        kernel_factory = MWGKernelFactory(config=MWGConfig(MALAKernelFactory(config=MALAConfig(0.1, C=None)))),
        # kernel_factory=HMCKernelFactory(HMCConfig(step_size=0.1, inverse_mass_matrix=imm, num_integration_steps=1)),
        # kernel_factory=NUTSKernelFactory(NUTSConfig(step_size=0.1, inverse_mass_matrix=imm)),
        num_samples=num_particles,
        num_chains=num_particles,
        # thinning_factor=100,
        thinning_factor=2,
        # thinning_factor=1,
        num_warmup_steps=num_mala_steps,
        adapt_step_size=True,
        init_using_log_l_mode=False
        # target_accept_rate=0.2,
    )
    config_first_iter = config.replace(num_warmup_steps=100)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist

def _get_default_train_nuts_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str
) -> _r_nuts:
    if ebm_model_type == "likelihood":
        imm = jnp.ones((task_config.dim_data,))
    else:
        imm = jnp.ones((task_config.dim_data + task_config.dim_parameters,))

    config=MCMCConfig(
        kernel_factory=NUTSKernelFactory(config=NUTSConfig(0.1, None, max_tree_depth=5)),
        # kernel_factory=HMCKernelFactory(HMCConfig(step_size=0.1, inverse_mass_matrix=imm, num_integration_steps=1)),
        # kernel_factory=NUTSKernelFactory(NUTSConfig(step_size=0.1, inverse_mass_matrix=imm)),
        num_samples=1000,
        num_chains=1000,
        thinning_factor=10,
        num_warmup_steps=num_mala_steps,
        adapt_mass_matrix=False,
        adapt_step_size=True,
        init_using_log_l_mode=False
    )
    config_first_iter = config.replace(num_warmup_steps=500)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist


def _get_default_train_adaptive_mala_config(
    task_config: TaskConfig, num_mala_steps: int, proposal: Literal["data", "noise", "prior+noise"], ebm_model_type: str
) -> _r_adaptive_mala:
    config=MCMCConfig(
        kernel_factory=AdaptiveMALAKernelFactory(config=AdaptiveMALAConfig(0.1)),
        num_samples=1000,
        num_chains=1000,
        thinning_factor=1,
        num_warmup_steps=num_mala_steps,
        adapt_mass_matrix=False,
        adapt_step_size=False,
        warmup_method="sbi_ebm"
    )
    config_first_iter = config.replace(num_warmup_steps=500)
    # config=MCMCConfig(
    #     kernel_factory=AdaptiveMALAKernelFactory(config=AdaptiveMALAConfig(0.1)),
    #     num_samples=1000,
    #     num_chains=10,
    #     thinning_factor=10,
    #     num_warmup_steps=10,
    #     adapt_mass_matrix=False,
    #     adapt_step_size=False
    # )
    # config_first_iter = config.replace(num_warmup_steps=500)
    init_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (MCMCAlgorithmFactory(config_first_iter), MCMCAlgorithmFactory(config)), init_dist



def _get_default_train_smc_config(
    num_particles: int, task_config: TaskConfig, num_smc_steps: int, num_mala_steps: int,
    ess_threshold: float, use_nuts: bool, total_dim: int, proposal: Literal["data", "noise", "prior+noise"],
    type_: Literal["smc", "smc:nonadaptive", "smc:blackjax", "smc:gibbs"], ebm_model_type: str
) -> _r_smc:
    if type_ == "smc:gibbs":
        inner_kernel_factory = DiscreteGibbsKernelFactory(
            DiscreteGibbsConfig()
        )
        factory = SMCFactory(SMCConfig(
            num_samples=1000,
            ess_threshold=ess_threshold,
            inner_kernel_factory=inner_kernel_factory,
            num_steps=num_smc_steps,
            inner_kernel_steps=num_mala_steps,
            record_trajectory=True
        ))
    else:
        # inner_kernel_factory = MALAKernelFactory(MALAConfig(0.1))
        # inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1, C=jnp.ones((total_dim,))))
        # inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1, update_cov=True, use_dense_cov=False))
        # inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1, update_cov=True, use_dense_cov=False))
        # inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1, update_cov=False, use_dense_cov=False))
        # inner_kernel_factory = MALAKernelFactory(MALAConfig(0.07))
        from sbi_ebm.samplers.kernels.numpyro_nuts import NUTS, NUTSConfig, NUTSKernelFactory
        # inner_kernel_factory = NUTSKernelFactory(NUTSConfig(step_size=0.1, C=jnp.ones((task_config.dim_data + task_config.dim_parameters,)), max_tree_depth=2))
        if type == "smc:nonadaptive":
            inner_kernel_factory = MALAKernelFactory(MALAConfig(0.08))
            factory = SMCFactory(SMCConfig(
                num_samples=num_particles,
                ess_threshold=ess_threshold,
                inner_kernel_factory=inner_kernel_factory,
                num_steps=num_smc_steps,
                inner_kernel_steps=num_mala_steps,
                # num_step_sizes=100
            ))
        else:
            inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1, update_cov=False, use_dense_cov=False))
            factory = SMCFactory(SMCConfig(
                num_samples=num_particles,
                ess_threshold=ess_threshold,
                inner_kernel_factory=inner_kernel_factory,
                num_steps=num_smc_steps,
                inner_kernel_steps=num_mala_steps,
                # num_step_sizes=100
            ))
        # if isinstance(inner_kernel_factory, NUTSKernelFactory):
        #     num_mala_steps = 1

        # factory = AdaptiveSMCFactory(AdaptiveSMCConfig(
        # factory = SMCFactory(SMCConfig(
        #     num_samples=1000,
        #     ess_threshold=ess_threshold,
        #     inner_kernel_factory=inner_kernel_factory,
        #     num_steps=num_smc_steps,
        #     inner_kernel_steps=num_mala_steps,
        # ))
    factory_first_iter = factory.replace(config=factory.config.replace(num_steps=10))

    assert ebm_model_type == "joint_tilted"
    init_smc_dist = _make_train_sampler_init_dist(task_config, proposal, ebm_model_type)
    return (factory_first_iter, factory), init_smc_dist


def _get_default_inference_mala_config(
    task_config: TaskConfig, num_posterior_samples: int, proposal: Literal["prior", "noise"]
) -> InferenceConfig:
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=MALAKernelFactory(config=MALAConfig(0.01)),
            num_samples=num_posterior_samples,
            num_chains=100,
            thinning_factor=10,
            num_warmup_steps=5000,
            adapt_mass_matrix=False,
            adapt_step_size=True
        )
    )
    posterior_sampling_init_dist = np_distributions.MultivariateNormal(
        jnp.zeros((task_config.dim_parameters,)), jnp.eye(task_config.dim_parameters)
    )
    posterior_sampling_init_dist = task_config.prior
    ret = InferenceConfig(
        num_samples=num_posterior_samples,
        sampling_config=config,
        sampling_init_dist=posterior_sampling_init_dist,
        should_z_score=(proposal == "prior")
    )
    return ret



def _get_default_inference_ula_config(
    task_config: TaskConfig, num_posterior_samples: int, proposal: Literal["prior", "noise"]
) -> InferenceConfig:
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=ULAKernelFactory(config=ULAConfig(0.01)),
            num_samples=num_posterior_samples,
            num_chains=100,
            thinning_factor=10,
            num_warmup_steps=1000,
            adapt_mass_matrix=False,
            adapt_step_size=True
        )
    )
    posterior_sampling_init_dist = np_distributions.MultivariateNormal(
        jnp.zeros((task_config.dim_parameters,)), jnp.eye(task_config.dim_parameters)
    )
    posterior_sampling_init_dist = task_config.prior
    ret = InferenceConfig(
        num_samples=num_posterior_samples,
        sampling_config=config,
        sampling_init_dist=posterior_sampling_init_dist,
        should_z_score=(proposal == "prior")
    )
    return ret

def _get_default_inference_smc_config(
    task_config: TaskConfig, num_posterior_samples: int, proposal: Literal["prior", "noise"]
) -> InferenceConfig:
    inner_kernel_factory = MALAKernelFactory(MALAConfig(0.01))
    config = SMCFactory(config=SMCConfig(
        num_samples=num_posterior_samples,
        ess_threshold=0.8,
        inner_kernel_factory=inner_kernel_factory,
        num_steps=3000,
        inner_kernel_steps=3,
        record_trajectory=False
    ))
    # config = AdaptiveSMCFactory(config=AdaptiveSMCConfig(
    #     num_samples=num_posterior_samples,
    #     ess_threshold=0.8,
    #     inner_kernel_factory=inner_kernel_factory,
    #     num_steps=1000,
    #     inner_kernel_steps=3,
    # ))
    posterior_sampling_init_dist = _make_inference_sampler_init_dist(task_config, proposal)

    ret = InferenceConfig(
        num_samples=num_posterior_samples,
        sampling_config=config,
        sampling_init_dist=posterior_sampling_init_dist,
        should_z_score=(proposal == "prior")
    )
    return ret


def _get_default_inference_savm_config(
    task_config: TaskConfig, num_posterior_samples: int, proposal: Literal["prior", "noise"],
    inner_sampler_num_steps: int = 100, num_warmup_steps: int = 2000
) -> InferenceConfig:
    config = MCMCAlgorithmFactory(
        config=MCMCConfig(
            kernel_factory=SAVMKernelFactory(config=SAVMConfig(
                aux_var_kernel_factory=MALAKernelFactory(MALAConfig(0.1)),
                # aux_var_num_inner_steps=500,
                # aux_var_num_inner_steps=500,  # examples/experiments/slurm-logs/run-TEST_20_rounds-165757070536.sbatch-2450128.out
                aux_var_num_inner_steps=inner_sampler_num_steps,
                base_var_kernel_factory=RWKernelFactory(config=RWConfig(0.1, jnp.ones((task_config.dim_parameters,)))),
                aux_var_init_strategy="x_obs"
            )),
            num_samples=num_posterior_samples,
            num_chains=100,
            thinning_factor=20,
            num_warmup_steps=num_warmup_steps,
            adapt_mass_matrix=False,
            adapt_step_size=True,
            progress_bar=True,
            target_accept_rate=0.2,
            # target_accept_rate=0.2  # 2488779,80
            # target_accept_rate=0.5
            init_using_log_l_mode=False
        )
    )
    posterior_sampling_init_dist = np_distributions.MultivariateNormal(
        jnp.zeros((task_config.dim_parameters,)), jnp.eye(task_config.dim_parameters)
    )
    posterior_sampling_init_dist = task_config.prior
    ret = InferenceConfig(
        num_samples=num_posterior_samples,
        sampling_config=config,
        sampling_init_dist=posterior_sampling_init_dist,
        should_z_score=(proposal == "prior")
    )
    return ret


def _make_likelihood_estimation_config(
    task_config: TaskConfig, num_smc_steps: int, num_mala_steps: int,
    enabled, ebm_model_type: str
) -> LikelihoodEstimationConfig:
    if enabled:
        assert ebm_model_type == "joint_tilted"
    if task_config.task_name == "Lorenz96":
        # slightly reduce computational time for high-dim datasets
        num_particles = 1000
    else:
        num_particles = 3000

    init_dist = _make_train_sampler_init_dist(task_config, "prior+noise", "joint_tilted")
    assert isinstance(init_dist, np_distributions.Distribution)
    inner_kernel_factory = MALAKernelFactory(MALAConfig(0.035))
    # inner_kernel_factory = AdaptiveMALAKernelFactory(AdaptiveMALAConfig(0.1))
    config = SMCFactory(SMCConfig(
        num_samples=1000,
        ess_threshold=0.5,
        inner_kernel_factory=inner_kernel_factory,
        num_steps=num_smc_steps,
        inner_kernel_steps=min(num_mala_steps, 5),
    ))
    # inner_kernel_factory = MALAKernelFactory(MALAConfig(0.1))
    # config = AdaptiveSMCFactory(AdaptiveSMCConfig(
    #     num_samples=num_particles,
    #     ess_threshold=0.8,
    #     inner_kernel_factory=inner_kernel_factory,
    #     num_steps=num_smc_steps,
    #     inner_kernel_steps=min(num_mala_steps, 5),
    # ))
    return LikelihoodEstimationConfig(
        enabled=enabled,
        num_particles=num_particles,
        alg=config,
        use_warm_start=True,
        init_dist=init_dist,
    )


def _resolve_configs(
    task_config: TaskConfig,
    num_samples: int,
    max_iter: int,
    learning_rate: float,
    weight_decay: float,
    gradient_penalty_val: float,
    num_smc_steps: int,
    num_mala_steps: int,
    num_particles: int,
    ess_threshold: float,
    use_warm_start: bool,
    num_posterior_samples: int,
    sampler: Literal["mala", "smc", "smc:nonadaptive", "ula", "adaptive_mala"],
    restart_every: Optional[int] = None,
    use_nuts: bool = False,
    noise_injection_val: float = 0.,
    proposal: Literal["noise", "data", "prior+noise"] = "noise",
    inference_sampler: Literal["mala", "smc", "ula", "adaptive_mala", "exchange_mcmc"] = "smc",
    batch_size: Optional[int] = None,
    select_based_on_test_loss: bool = True,
    ebm_model_type: Literal["joint_tilted", "joint_unbiased", "likelihood", "ratio"] = "joint_tilted",
    estimate_loss: bool = True,
    inference_proposal: Literal["prior", "noise"] = "prior",
    ebm_depth: Union[int, Literal['auto']] = 'auto',
    ebm_width: Union[int, Literal['auto']] = 'auto',
    exchange_mcmc_inner_sampler_num_steps: int = 100,
    inference_num_warmup_steps: int = 2000,
    training_num_frozen_steps: int = 50,
) -> Tuple[TrainingConfig, InferenceConfig]:

    optimizer_config = OptimizerConfig(
        learning_rate=learning_rate, weight_decay=weight_decay,
        noise_injection_val=noise_injection_val
    )

    if task_config.task_name in (
            "bernoulli_glm_raw", "Lorenz96",
            "ornstein_uhlenbeck"
    ):
        ebm_config = EBMLikelihoodConfig(width=50)
    elif task_config.task_name == "pyloric":
        # ebm_config = EBMLikelihoodConfig(width=300, depth=6) examples/experiments/slurm-logs/run-TEST_20_rounds-165757070536.sbatch-2450128.out
        ebm_config = EBMLikelihoodConfig(width=300, depth=6)
    else:
        ebm_config = EBMLikelihoodConfig()

    if ebm_depth != 'auto':
        ebm_config = ebm_config.replace(depth=ebm_depth)
    if ebm_width != 'auto':
        ebm_config = ebm_config.replace(width=ebm_width)



    print(f'using a network of width {ebm_config.width} and depth {ebm_config.depth}')

    likelihood_estimation_config = _make_likelihood_estimation_config(
        task_config, num_smc_steps, num_mala_steps, enabled=estimate_loss, ebm_model_type=ebm_model_type
    )

    def _make_training_config(
        cfg_first_iter: InferenceAlgorithmFactory, cfg: InferenceAlgorithmFactory,
        init_dist: Union[np_distributions.Distribution, Literal["data"]]
    ):
        return TrainingConfig(
            max_iter=max_iter,
            ebm=ebm_config,
            gradient_penalty_val=gradient_penalty_val,
            sampling_cfg_first_iter=cfg_first_iter,
            sampling_cfg=cfg,
            sampling_init_dist=init_dist,
            likelihood_estimation_config=likelihood_estimation_config,
            num_particles=num_particles,
            use_warm_start=use_warm_start,
            optimizer=optimizer_config,
            restart_every=restart_every,
            batch_size=batch_size,
            batching_enabled=batch_size is not None,
            select_based_on_test_loss=select_based_on_test_loss,
            ebm_model_type=ebm_model_type
        )

    if sampler == "gibbs":
        assert ebm_model_type == "ratio"
        (cfg_first_iter, cfg), init_dist = _get_default_train_gibbs_sampling_config(
            task_config, num_mala_steps, proposal=proposal, ebm_model_type=ebm_model_type
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "smc" or sampler == "smc:nonadaptive" or sampler == "smc:blackjax" or sampler == "smc:gibbs":
        if sampler == "smc:gibbs":
            assert ebm_model_type == "ratio"
        (cfg_first_iter, cfg), init_dist = _get_default_train_smc_config(
            num_particles,
            task_config, num_smc_steps, num_mala_steps, ess_threshold,
            use_nuts=use_nuts,
            total_dim=task_config.prior.event_shape[0] + task_config.x_obs.shape[0],
            proposal=proposal, type_=sampler, ebm_model_type=ebm_model_type
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "adaptive_mala":
        (cfg_first_iter, cfg), init_dist = _get_default_train_adaptive_mala_config(
            task_config, num_mala_steps, proposal, ebm_model_type=ebm_model_type
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "mala":
        (cfg_first_iter, cfg), init_dist = _get_default_train_mala_config(
            task_config, num_mala_steps, proposal, ebm_model_type=ebm_model_type,
            num_frozen_steps=training_num_frozen_steps
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "mwg":
        (cfg_first_iter, cfg), init_dist = _get_default_train_mwg_config(
            task_config, num_mala_steps, proposal, ebm_model_type=ebm_model_type,
            num_particles=num_particles
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "nuts":
        (cfg_first_iter, cfg), init_dist = _get_default_train_nuts_config(
            task_config, num_mala_steps, proposal, ebm_model_type=ebm_model_type
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    elif sampler == "ula":
        (cfg_first_iter, cfg), init_dist = _get_default_train_ula_config(
            task_config, num_mala_steps, proposal, ebm_model_type=ebm_model_type
        )
        train_config = _make_training_config(cfg_first_iter, cfg, init_dist)

    else:
        raise ValueError

    if inference_sampler == "smc":
        inference_config = _get_default_inference_smc_config(
            task_config, num_posterior_samples, inference_proposal
        )
    elif inference_sampler == "mala":
        inference_config = _get_default_inference_mala_config(
            task_config, num_posterior_samples, inference_proposal
        )
    elif inference_sampler == "ula":
        inference_config = _get_default_inference_ula_config(
            task_config, num_posterior_samples, inference_proposal
        )
    elif inference_sampler == "exchange_mcmc":
        inference_config = _get_default_inference_savm_config(
            task_config, num_posterior_samples, inference_proposal,
            inner_sampler_num_steps=exchange_mcmc_inner_sampler_num_steps,
            num_warmup_steps=inference_num_warmup_steps

        )
    else:
        raise ValueError

    if train_config.num_particles is None:
        train_config = train_config.replace(num_particles=num_samples)
    return train_config, inference_config



class MetricResults(NamedTuple):
    mmd: Number
    c2st: Number


class TrainEvalTresults(NamedTuple):
    train_results: Results
    eval_results: Optional[MetricResults] = None


def _evaluate_posterior(
    posterior_samples: Array, task: Task, num_observation: int
) -> MetricResults:
    from sbibm.metrics import c2st, mmd

    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)
    sbi_ebm_posterior_samples = torch.from_numpy(np.array(posterior_samples))

    mmd_val = mmd(reference_posterior_samples, sbi_ebm_posterior_samples).item()
    c2st_val = c2st(reference_posterior_samples, sbi_ebm_posterior_samples)[0].item()
    return MetricResults(mmd_val, c2st_val)


def _make_args_hashable(args: dict):
    import copy
    hashable_args = copy.deepcopy(args)
    task = hashable_args["task"]
    assert isinstance(task, Task) or isinstance(task, str)
    hashable_args["task"] = task.name if isinstance(task, Task) else task

    # XXX
    hashable_args.pop("single_round_results", None)

    # ensure that args are json-serializable
    import json
    _ = json.dumps(hashable_args)
    return hashable_args


def _custom_hash(obj: Dict) -> str:
    # build a filename by hashing the config dict
    import hashlib
    import json
    hash_str = hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
    return hash_str


def make_checkpoint_path(config_dict: Dict[str, Any]) -> str:
    from pathlib import Path
    import os
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    if slurm_job_name is not None:
        checkpoint_dir = Path(os.getcwd()) / "checkpoints" / slurm_job_name
    else:
        checkpoint_dir = Path(os.getcwd()) / "checkpoints" / "noslurm"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = _custom_hash(config_dict)

    # TODO: de-hardcode the directory where the checkpoints are stored
    checkpoint_filepath = checkpoint_dir / f"{checkpoint_filename}.pkl"
    return str(checkpoint_filepath)


def resolve_checkpointing_config(
    config_dict: Dict[str, Any],
    start_from_checkpoint: Union[bool, str],
    init_checkpoint_round: int,
    checkpoint: Union[bool, str],
) -> CheckpointConfig:

    should_start_from_checkpoint = start_from_checkpoint is True or isinstance(start_from_checkpoint, str)
    if start_from_checkpoint is True:
        # if no specific chekpoint path is provided, resolve the checkpoint
        # path without the `start_from_checkpoint` argument since otherwise the checkpoint path
        # will not be the same between the first call (where start_from_checkpoint can be False)
        # and subsequent calls (where start_from_checkpoint can be True)
        # XXX: maybe instead I should allow for specifying start_from_checkpoint=True even when calling
        # the run function for the first time, when there is no checkpoints yet.
        init_checkpoint_path = make_checkpoint_path({k: v for k, v in config_dict.items() if k not in ("start_from_checkpoint", "init_checkpoint_round")})
    elif isinstance(start_from_checkpoint, str):
        init_checkpoint_path = start_from_checkpoint
    else:
        init_checkpoint_path = ''

    should_checkpoint = checkpoint is True or isinstance(checkpoint, str)
    if checkpoint is True:
        # ditto
        checkpoint_path = make_checkpoint_path({k: v for k, v in config_dict.items() if k not in ("start_from_checkpoint", "init_checkpoint_round")})
    elif isinstance(checkpoint, str):
        checkpoint_path = checkpoint
    else:
        checkpoint_path = ''

    if should_start_from_checkpoint:
        print(f"Starting from checkpoint {init_checkpoint_path}")

    if should_checkpoint:
        print(f"Checkpointing to {checkpoint_path}")

    checkpoint_config = CheckpointConfig(
        should_start_from_checkpoint=should_start_from_checkpoint,
        init_checkpoint_path=init_checkpoint_path,
        init_checkpoint_round=init_checkpoint_round,
        should_checkpoint=should_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    return checkpoint_config


def run(
    task: Union[Task, str],
    num_samples: Union[int, Tuple[int, ...]],
    num_observation: int,
    max_iter: int = 3000,
    learning_rate: float = 0.001,
    weight_decay: float = 0.1,
    gradient_penalty_val: float = 0,
    num_smc_steps: int = 30,
    num_mala_steps: int = 3,
    num_particles: int = 1000,
    ess_threshold: float = 0.8,
    use_warm_start=True,
    num_posterior_samples=10000,
    random_seed=41,
    sampler: Literal["mala", "adaptive_mala", "smc", "ula"] = "smc",
    evaluate_posterior: bool = False,
    restart_every: Optional[int] = None,
    init_proposal: str = "prior",
    use_nuts=False,
    tempering_coef: Optional[int] = None,
    noise_injection_val: float = 0.001,
    proposal: Literal["noise", "data", "prior+noise"] = "noise",
    inference_sampler: Literal["mala", "adaptive_mala", "smc", "ula", "exchange_mcmc"] = "smc",
    batch_size: Optional[int] = 1000,
    select_based_on_test_loss: bool = True,
    use_data_from_past_rounds: bool = True,
    discard_prior_samples: bool = False,
    ebm_model_type: Literal["joint_tilted", "joint_unbiased", "likelihood", "ratio"] = "joint_tilted",
    fit_in_unconstrained_space: bool = False,
    start_from_checkpoint: Union[bool, str] = False,
    init_checkpoint_round: int = -1,
    checkpoint: bool = False,
    estimate_loss: Union[bool, Literal["auto"]] = "auto",
    inference_proposal: Literal["prior", "noise"] = "noise",
    single_round_results: Optional[Union[List[SingleRoundResults], Tuple[SingleRoundResults]]] = None,
    ebm_depth: Union[int, Literal['auto']] = 'auto',
    ebm_width: Union[int, Literal['auto']] = 'auto',
    exchange_mcmc_inner_sampler_num_steps: int = 100,
    inference_num_warmup_steps: int = 2000,
    n_sigma: float = 3.0,
    training_num_frozen_steps: int = 50,
):
    # some args validation
    if not fit_in_unconstrained_space:
        assert inference_proposal == "prior"
        assert proposal in ("data", "prior+noise")

    checkpointing_config = resolve_checkpointing_config(
        _make_args_hashable(locals()), start_from_checkpoint, init_checkpoint_round, checkpoint
    )
    if single_round_results is not None:
        checkpointing_config = CheckpointConfig(
            should_start_from_checkpoint=True,
            init_checkpoint_path=None,  # type: ignore
            init_checkpoint_round=init_checkpoint_round,
            should_checkpoint=False,
            checkpoint_path=None,  # type: ignore
            single_round_results=list(single_round_results)
        )


    if isinstance(task, str):
        task = get_task(task)
    assert isinstance(task, Task)

    if isinstance(num_samples, int):
        num_samples = (num_samples,)
    num_rounds = len(num_samples)
    assert num_rounds > 0

    seed_everything(random_seed)
    key = random.PRNGKey(random_seed)

    use_calibration_kernel = task.name == "pyloric"
    task_config = TaskConfig.from_task(
        JaxTask(task), num_observation=num_observation,
        use_calibration_kernel=use_calibration_kernel
    )

    if init_proposal == "prior":
        proposal_dist = task_config.prior
    elif init_proposal == "reference_posterior":
        proposal_dist = get_reference_posterior(JaxTask(task), num_observation)
    else:
        from pathlib import Path
        import cloudpickle
        file_, round = init_proposal.split(":")
        file_, round = Path(file_), int(round)
        assert file_.exists()
        print(f"loading proposal {round} from file {file_}...")

        with open(file_, "rb") as f:
            results: TrainEvalTresults = cloudpickle.load(f)
        proposal_dist = results.train_results.single_round_results[round].dataset.prior
        if isinstance(proposal_dist, KDEApproxDist):
            proposal_dist = proposal_dist._dist

    if tempering_coef is not None:
        proposal_config = ProposalConfig(
            init_proposal=proposal_dist,
            tempering_enabled=True,
            mog_config=MOGTrainingConfig(
                num_clusters=50, min_std=tempering_coef,
                max_iter=100, num_inits=4,
                cov_reg_param=tempering_coef # NEW
            ),
            t_prior=0.
        )
    else:
        proposal_config = ProposalConfig(
            init_proposal=proposal_dist,
            tempering_enabled=False
        )


    if estimate_loss == "auto":
        estimate_loss = (ebm_model_type not in ("likelihood", "joint_unbiased"))

    if ebm_model_type in ("joint_unbiased", "likelihood"):
        assert not estimate_loss, "loss estimation is not supported for this model type"
        assert inference_sampler == "exchange_mcmc", "only exchange_mcmc is supported for this model type"

    if ebm_model_type in ("joint_unbiased", "likelihood") or not estimate_loss:
        select_based_on_test_loss = False

    training_config, inference_config = _resolve_configs(
        task_config,
        num_samples[0],
        max_iter,
        learning_rate,
        weight_decay,
        gradient_penalty_val,
        num_smc_steps,
        num_mala_steps,
        num_particles,
        ess_threshold,
        use_warm_start,
        num_posterior_samples,
        sampler,
        restart_every,
        use_nuts,
        noise_injection_val=noise_injection_val,
        proposal=proposal,
        inference_sampler=inference_sampler,
        batch_size=batch_size,
        select_based_on_test_loss=select_based_on_test_loss,
        ebm_model_type=ebm_model_type,
        estimate_loss=estimate_loss,
        inference_proposal=inference_proposal,
        ebm_depth=ebm_depth,
        ebm_width=ebm_width,
        exchange_mcmc_inner_sampler_num_steps=exchange_mcmc_inner_sampler_num_steps,
        inference_num_warmup_steps=inference_num_warmup_steps,
        training_num_frozen_steps=training_num_frozen_steps,
    )

    def _make_config(training_config: TrainingConfig, inference_config: InferenceConfig) -> Config:
        return Config(
            num_samples=num_samples,
            training=training_config,
            task=task_config,
            inference=inference_config,
            proposal=proposal_config,
            use_data_from_past_rounds=use_data_from_past_rounds,
            discard_prior_samples=discard_prior_samples,
            preprocessing=PreProcessingConfig(biject_to_unconstrained_space=fit_in_unconstrained_space),
            checkpointing=checkpointing_config,
            n_sigma=n_sigma,
        )

    # Could not find a better way to make the type checker happy...
    # SMC training
    c = _make_config(training_config, inference_config)
    if c.training.ebm_model_type == "likelihood":
        trainer = LikelihoodTrainer()
    else:
        trainer = Trainer()
    m = MultiRoundTrainer(trainer)
    train_results = m.train_sbi_ebm(c, key=key)

    if evaluate_posterior:
        eval_results = _evaluate_posterior(
            train_results.posterior_samples, task, num_observation
        )
    else:
        eval_results = None

    return TrainEvalTresults(train_results, eval_results)
