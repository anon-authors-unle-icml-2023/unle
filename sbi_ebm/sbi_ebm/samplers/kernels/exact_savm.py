from typing import Generic, Literal, NamedTuple, Optional, Tuple, Type, cast
from jax.core import Tracer

import jax.numpy as jnp
from flax import struct
from jax import random, vmap
from jax.lax import scan  # type: ignore
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions
from typing_extensions import Self

from numpyro.infer import BarkerMH

from sbi_ebm.distributions import (DoublyIntractableLogDensity,
                                   ThetaConditionalLogDensity)
from sbi_ebm.pytypes import Array, DoublyIntractableLogDensity_T, LogDensity_T, Numeric, PRNGKeyArray
from sbi_ebm.samplers.kernels.base import (Array_T, Config_T, Info, Info_T, Kernel, KernelConfig,
                                           KernelFactory, MHKernel, MHKernelFactory, Result, State, State_T, TunableKernel, TunableMHKernelFactory)
# from sbi_ebm.samplers.mala import MALAConfig, MALAKernel, _mala
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAInfo, MALAKernel, MALAKernelFactory, MALAState
from sbi_ebm.samplers.kernels.rwmh import RWInfo, RWKernel, RWKernelFactory, RWState

from jax.nn import logsumexp, softmax


class DiscretizingSampler(struct.PyTreeNode):
    log_prob: LogDensity_T
    bounds: Tuple[Tuple[int, int], Tuple[int, int]] = ((-10, 10), (-10, 10))
    nbins: int = 100


    def sample(self, key: PRNGKeyArray) -> Array_T:
        (x_min, x_max), (y_min, y_max) = self.bounds

        num_total_points = self.nbins ** len(self.bounds)

        _X, _Y = jnp.meshgrid(
            jnp.linspace(x_min, x_max, self.nbins), jnp.linspace(y_min, y_max, self.nbins),
            indexing="ij",
        )
        _inputs = jnp.stack((_X, _Y), axis=-1).reshape(num_total_points, 2)  # type: ignore
        conditioned_log_density_vals = vmap(self.log_prob)(_inputs)


        key, subkey = random.split(key)
        idx = random.choice(subkey, len(conditioned_log_density_vals), p = softmax(conditioned_log_density_vals))  # type: ignore
        return _inputs[idx]





class ExactSAVMConfig(Generic[Config_T, State_T, Info_T], KernelConfig):
    base_var_kernel_factory: RWKernelFactory


class ExactSAVMInfo(Generic[Info_T], Info):
    accept: bool
    log_alpha: float


class ExactSAVMState(Generic[Config_T, State_T, Info_T], State):
    base_var_state: RWState = struct.field(pytree_node=True)
    aux_var: Array


class ExactSAVMResult(NamedTuple):
    x: ExactSAVMState
    accept_freq: Numeric


class ExactSAVMKernel(TunableKernel[ExactSAVMConfig, ExactSAVMState, ExactSAVMInfo], Generic[Config_T, State_T, Info_T]):
    target_log_prob: DoublyIntractableLogDensity
    config: ExactSAVMConfig[Config_T, State_T, Info_T]

    @property
    def base_var_kernel(self):
        return self.config.base_var_kernel_factory.build_kernel(self.target_log_prob)

    def get_step_size(self) -> Numeric:
        return self.config.base_var_kernel_factory.config.step_size

    def get_inverse_mass_matrix(self) -> Numeric:
        C = self.config.base_var_kernel_factory.config.C
        assert C is not None
        return C

    def set_step_size(self, step_size) -> Self:
        return self.replace(config=self.config.replace(base_var_kernel_factory=self.config.base_var_kernel_factory.replace(config=self.base_var_kernel.set_step_size(step_size).config)))

    def set_inverse_mass_matrix(self, inverse_mass_matrix) -> Self:
        base_var_kernel = self.config.base_var_kernel_factory.build_kernel(self.target_log_prob)
        return self.replace(config=self.config.replace(base_var_kernel_factory=self.config.base_var_kernel_factory.replace(config=self.base_var_kernel.set_inverse_mass_matrix(inverse_mass_matrix).config)))

    @classmethod
    def create(cls: Type[Self], target_log_prob: DoublyIntractableLogDensity, config: ExactSAVMConfig[Config_T, State_T, Info_T]) -> Self:
        return cls(target_log_prob, config)

    def init_state(self: Self, x: Array_T, aux_var0: Optional[Array_T] = None) -> ExactSAVMState[Config_T, State_T, Info_T]:
        assert len(self.target_log_prob.x_obs.shape) == 1

        if aux_var0 is None:
            resolved_aux_var0 = jnp.zeros_like(self.target_log_prob.x_obs)
        else:
            resolved_aux_var0 = aux_var0

        init_log_l = ThetaConditionalLogDensity(self.target_log_prob.log_likelihood, x)
        aux_var = self.target_log_prob.x_obs

        # x: theta
        base_var_state = self.base_var_kernel.init_state(x)
        return ExactSAVMState(base_var_state.x, base_var_state, aux_var)

    def _build_info(self, accept: bool, log_alpha: Numeric) -> ExactSAVMInfo[Info_T]:
        return ExactSAVMInfo(accept, log_alpha)

    def _sample_from_proposal(
        self, key: PRNGKeyArray, state: ExactSAVMState[Config_T, State_T, Info_T]
    ) -> ExactSAVMState:
        key, key_base_var, key_aux_var = random.split(key, num=3)

        # first, sample base variable
        new_base_var_state = self.base_var_kernel._sample_from_proposal(key_base_var, state.base_var_state)

        this_iter_log_l = ThetaConditionalLogDensity(
            self.target_log_prob.log_likelihood, new_base_var_state.x
        )

        key, subkey = random.split(key)
        new_x = DiscretizingSampler(this_iter_log_l).sample(subkey)


        return state.replace(x=new_base_var_state.x, base_var_state=new_base_var_state, aux_var=new_x)

    def _compute_accept_prob(
        self,
        proposal: ExactSAVMState[Config_T, State_T, Info_T],
        x: ExactSAVMState[Config_T, State_T, Info_T],
    ) -> Numeric:
        """Compute α = min(1, (p(xᵢ₊₁)q(xᵢ | xᵢ₊₁)) / (p(xᵢ) q(xᵢ₊₁ | xᵢ)))"""
        # orig_x = theta
        q_theta = self.base_var_kernel.get_proposal()
        log_q_new_given_prev = q_theta.log_prob(
            x=proposal.base_var_state.x, x_cond=x.base_var_state.x
        )
        log_q_prev_given_new = q_theta.log_prob(
            x=x.base_var_state.x, x_cond=proposal.base_var_state.x
        )

        log_alpha = (
            self.target_log_prob(proposal.base_var_state.x)
            + log_q_prev_given_new
            - self.target_log_prob(x.base_var_state.x)
            - log_q_new_given_prev
            + self.target_log_prob.log_likelihood(
                x.base_var_state.x, proposal.aux_var
            )
            - self.target_log_prob.log_likelihood(
                proposal.base_var_state.x, proposal.aux_var
            )
        )
        log_alpha = jnp.nan_to_num(log_alpha, nan=-50, neginf=-50, posinf=0)

        return log_alpha

    # def one_step(self, x: ExactSAVMState[State_T, Info_T], key: PRNGKeyArray) -> Result[ExactSAVMState[State_T, Info_T], ExactSAVMInfo]:
    #     ret = cast(ExactSAVMState[State_T, Info_T], super(ExactSAVMKernel, self).one_step(x, key))
    #     return ret.replace(info=ret.info.replace(aux_var_info=ret.state.aux_var_info))


class ExactSAVMKernelFactory(TunableMHKernelFactory[ExactSAVMConfig[Config_T, State_T, Info_T], ExactSAVMState[Config_T, State_T, Info_T], ExactSAVMInfo[Info_T]]):
    kernel_cls: Type[ExactSAVMKernel] = struct.field(pytree_node=False, default=ExactSAVMKernel)

    def build_kernel(self, log_prob: DoublyIntractableLogDensity) -> ExactSAVMKernel:
        return self.kernel_cls.create(log_prob, self.config)
