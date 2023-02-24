from typing import Any, Generic, Type
from typing import Callable, Optional, Tuple, Union
from typing_extensions import Self

from flax import struct
from jax import random, vmap
from jax.nn import logsumexp, softmax
import jax.numpy as jnp
from numpyro import distributions as np_distributions
from sbi_ebm.distributions import DoublyIntractableJointLogDensity, ThetaConditionalLogDensity, MixedJointLogDensity

from sbi_ebm.pytypes import Array, LogLikelihood_T, Numeric, PRNGKeyArray
from sbi_ebm.samplers.kernels.base import Config_T, Info, Info_T, KernelConfig, KernelFactory, MHKernel, MHKernelFactory, State, State_T, TunableConfig, TunableConfig_T
from sbi_ebm.samplers.kernels.discrete_gibbs import DiscreteLogDensity
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernel, MALAState
from sbi_ebm.samplers.kernels.numpyro_nuts import NUTSConfig
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation

from typing import Optional, Type, cast
from typing_extensions import Self

from numpyro.infer.hmc_util import HMCAdaptState, warmup_adapter
from sbi_ebm.samplers.kernels.base import Array_T, Info, KernelConfig, Result, State, TunableKernel, TunableMHKernelFactory
from sbi_ebm.pytypes import LogDensity_T, Numeric, PRNGKeyArray

from numpyro.infer.hmc import NUTS
from numpyro.infer.hmc_gibbs import HMCGibbs, HMCGibbsState as np_HMCGibbstate


import jax.numpy as jnp
from jax import random

from flax import struct

class MWGConfig(Generic[TunableConfig_T, State_T, Info_T], TunableConfig):
    continuous_kernel_factory: TunableMHKernelFactory[TunableConfig_T, State_T, Info_T] = struct.field(
        pytree_node=True
    )

    def get_step_size(self) -> Numeric:
        return self.continuous_kernel_factory.config.get_step_size()

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.continuous_kernel_factory.config.get_inverse_mass_matrix()

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(
            continuous_kernel_factory=self.continuous_kernel_factory.replace(
                config=self.continuous_kernel_factory.config.set_step_size(step_size)
            )
        )

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            continuous_kernel_factory=self.continuous_kernel_factory.replace(
                config=self.continuous_kernel_factory.config._set_dense_inverse_mass_matrix(inverse_mass_matrix)
            )
        )

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            continuous_kernel_factory=self.continuous_kernel_factory.replace(
                config=self.continuous_kernel_factory.config._set_diag_inverse_mass_matrix(inverse_mass_matrix)
            )
        )



class MWGInfo(Info):
    accept: Numeric
    log_alpha: Numeric


class MWGState(Generic[State_T], State):
    continuous_var_state: State_T


class MWGKernel(TunableKernel[MWGConfig[TunableConfig_T, State_T, Info_T], MWGState[State_T], MWGInfo]):
    config: MWGConfig[TunableConfig_T, State_T, Info_T]

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: MWGConfig
    ) -> Self:
        return cls(target_log_prob, config)

    def _sample_from_proposal(self, key: PRNGKeyArray, x: MWGState) -> MWGState:
        raise NotImplementedError

    def _compute_accept_prob(self, proposal: MWGState, x: MWGState) -> Numeric:
        raise NotImplementedError

    def get_step_size(self) -> Array_T:
        return self.config.continuous_kernel_factory.config.get_step_size()

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.continuous_kernel_factory.config.get_inverse_mass_matrix()

    def set_step_size(self, step_size) -> Self:
        return self.replace(
            config=self.config.replace(continuous_kernel_factory=self.config.continuous_kernel_factory.replace(
                config=self.config.continuous_kernel_factory.config.replace(step_size=step_size)
            ))
        )

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            config=self.config.replace(continuous_kernel_factory=self.config.continuous_kernel_factory.replace(
                config=self.config.continuous_kernel_factory.config.replace(C=inverse_mass_matrix)
            ))
        )

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            config=self.config.replace(continuous_kernel_factory=self.config.continuous_kernel_factory.replace(
                config=self.config.continuous_kernel_factory.config.replace(C=inverse_mass_matrix)
            ))
        )

    def init_state(self, x: Array_T) -> MWGState:
        assert isinstance(self.target_log_prob, MixedJointLogDensity)
        theta, x_continuous = x[..., : self.target_log_prob.dim_param], x[..., self.target_log_prob.dim_param :]
        log_likelihood = ThetaConditionalLogDensity(self.target_log_prob.log_likelihood, theta)
        x_kernel = self.config.continuous_kernel_factory.build_kernel(log_likelihood)
        x_continuous_init_state = x_kernel.init_state(x_continuous)
        return MWGState(x=x, continuous_var_state=x_continuous_init_state)


    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> MWGInfo:
        return MWGInfo(accept=accept, log_alpha=log_alpha)

    def one_step(self, x: MWGState, key: PRNGKeyArray) -> Result[MWGState, MWGInfo]:
        assert isinstance(self.target_log_prob, MixedJointLogDensity)
        key, subkey = random.split(key)
        prev_x = x.x[self.target_log_prob.dim_param:]
        new_theta = self.target_log_prob.sample_from_discrete_conditional(prev_x, key=subkey)

        log_likelihood = ThetaConditionalLogDensity(self.target_log_prob.log_likelihood, new_theta)

        x_kernel = self.config.continuous_kernel_factory.build_kernel(log_likelihood)
        key, subkey = random.split(key)
        x_result = x_kernel.n_steps(x.continuous_var_state, 10, subkey)

        info = MWGInfo(accept=x_result.info.accept, log_alpha=x_result.info.log_alpha)  # pyright: ignore [reportGeneralTypeIssues]
        new_state = MWGState(
            x=jnp.concatenate([new_theta, x_result.state.x]),
            continuous_var_state=x_result.state
        )
        return Result(new_state, info)



class MWGKernelFactory(TunableMHKernelFactory[MWGConfig, MWGState, MWGInfo]):
    kernel_cls: Type[MWGKernel] = struct.field(pytree_node=False, default=MWGKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> MWGKernel:
        return self.kernel_cls.create(log_prob, self.config)
