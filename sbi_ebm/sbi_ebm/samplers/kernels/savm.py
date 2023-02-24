from typing import Generic, Literal, Optional, Type, cast

import jax.numpy as jnp
from flax import struct
from jax import random
from typing_extensions import Self

from sbi_ebm.distributions import (DoublyIntractableLogDensity,
                                   ThetaConditionalLogDensity)
from sbi_ebm.pytypes import Numeric, PRNGKeyArray
from sbi_ebm.samplers.kernels.base import (Array_T, Config_T, Info, Info_T, 
                                           KernelFactory, Result, State, State_T, TunableConfig, TunableKernel, TunableMHKernelFactory)
# from sbi_ebm.samplers.mala import MALAConfig, MALAKernel, _mala
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.samplers.kernels.rwmh import RWConfig, RWInfo, RWKernelFactory, RWState


class SAVMConfig(Generic[Config_T, State_T, Info_T], TunableConfig):
    base_var_kernel_factory: RWKernelFactory = struct.field(pytree_node=True, default=RWKernelFactory(config=RWConfig()))
    aux_var_kernel_factory: KernelFactory[Config_T, State_T, Info_T] = struct.field(
        pytree_node=True, default=MALAKernelFactory(config=MALAConfig())
    )
    aux_var_num_inner_steps: int = struct.field(pytree_node=False, default=100)
    aux_var_init_strategy: Literal["warm", "x_obs"] = struct.field(pytree_node=False, default="warm")

    @property
    def supports_diagonal_mass(self) -> bool:
        return True

    def get_step_size(self) -> Array_T:
        return self.base_var_kernel_factory.config.get_step_size()

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.base_var_kernel_factory.config.get_inverse_mass_matrix()

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(
            base_var_kernel_factory=self.base_var_kernel_factory.replace(
                config=self.base_var_kernel_factory.config.set_step_size(step_size)
            )
        )

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            base_var_kernel_factory=self.base_var_kernel_factory.replace(
                config=self.base_var_kernel_factory.config._set_dense_inverse_mass_matrix(inverse_mass_matrix)
            )
        )

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(
            base_var_kernel_factory=self.base_var_kernel_factory.replace(
                config=self.base_var_kernel_factory.config._set_diag_inverse_mass_matrix(inverse_mass_matrix)
            )
        )


class SAVMInfo(Generic[Info_T], Info):
    accept: bool
    log_alpha: float
    theta_stats: RWInfo
    aux_var_info: Optional[Info_T] = None


class SAVMState(Generic[Config_T, State_T, Info_T], State):
    base_var_state: RWState = struct.field(pytree_node=True)
    aux_var_state: State_T = struct.field(pytree_node=True)
    kernel_config: Config_T = struct.field(pytree_node=True)
    aux_var_mcmc_chain: "_MCMCChain" = struct.field(pytree_node=True)  # pyright: ignore [reportUndefinedVariable]
    aux_var_info: Optional[Info_T] = struct.field(pytree_node=True, default=None)




class SAVMResult(Result[SAVMState[Config_T, State_T, Info_T], SAVMInfo[Info_T]], struct.PyTreeNode):
    x: SAVMState[Config_T, State_T, Info_T]
    accept_freq: Numeric


class SAVMKernel(TunableKernel[SAVMConfig[Config_T, State_T, Info_T], SAVMState[Config_T, State_T, Info_T], SAVMInfo[Info_T]]):
    target_log_prob: DoublyIntractableLogDensity
    config: SAVMConfig[Config_T, State_T, Info_T]

    @property
    def base_var_kernel(self):
        return self.config.base_var_kernel_factory.build_kernel(self.target_log_prob)

    def get_step_size(self) -> Numeric:
        return self.config.base_var_kernel_factory.config.step_size

    def get_inverse_mass_matrix(self) -> Numeric:
        C = self.config.base_var_kernel_factory.config.C
        assert C is not None
        return C

    def set_step_size(self, step_size: Array_T) -> Self:
        return self.replace(config=self.config.replace(base_var_kernel_factory=self.config.base_var_kernel_factory.replace(config=self.base_var_kernel.set_step_size(step_size).config)))

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(base_var_kernel_factory=self.config.base_var_kernel_factory.replace(config=self.base_var_kernel.set_inverse_mass_matrix(inverse_mass_matrix).config)))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(base_var_kernel_factory=self.config.base_var_kernel_factory.replace(config=self.base_var_kernel.set_inverse_mass_matrix(inverse_mass_matrix).config)))

    @classmethod
    def create(cls: Type[Self], target_log_prob: DoublyIntractableLogDensity, config: SAVMConfig[Config_T, State_T, Info_T]) -> Self:
        return cls(target_log_prob, config)

    def init_state(self: Self, x: Array_T, aux_var0: Optional[Array_T] = None) -> SAVMState[Config_T, State_T, Info_T]:
        assert len(self.target_log_prob.x_obs.shape) == 1

        if aux_var0 is None:
            resolved_aux_var0 = jnp.zeros_like(self.target_log_prob.x_obs)
        else:
            resolved_aux_var0 = aux_var0

        init_log_l = ThetaConditionalLogDensity(self.target_log_prob.log_likelihood, x)
        aux_var_kernel = self.config.aux_var_kernel_factory.build_kernel(init_log_l)
        aux_var_state = aux_var_kernel.init_state(resolved_aux_var0)

        # mcmc chain
        from sbi_ebm.samplers.inference_algorithms.mcmc.base import _MCMCChain, _MCMCChainConfig
        assert isinstance(aux_var_kernel, TunableKernel)
        tune_mass_matrix = aux_var_kernel.get_inverse_mass_matrix() is not None
        mcmc_chain = _MCMCChain(
            _MCMCChainConfig(
                self.config.aux_var_kernel_factory, self.config.aux_var_num_inner_steps // 2,
                False, self.config.aux_var_num_inner_steps // 2, True, tune_mass_matrix,
                init_using_log_l_mode=True, init_using_log_l_mode_num_opt_steps=50
            ),
            init_log_l, aux_var_state
        )

        # x: theta
        base_var_state = self.base_var_kernel.init_state(x)
        return SAVMState(base_var_state.x, base_var_state, aux_var_state, self.config.aux_var_kernel_factory.config, aux_var_mcmc_chain=mcmc_chain)

    def _build_info(self, accept: bool, log_alpha: Numeric) -> SAVMInfo[Info_T]:
        return SAVMInfo(accept, log_alpha, RWInfo(accept, log_alpha), None)

    def _sample_from_proposal(
        self, key: PRNGKeyArray, state: SAVMState[Config_T, State_T, Info_T]
    ) -> SAVMState[Config_T, State_T, Info_T]:
        key, key_base_var, key_aux_var = random.split(key, num=3)

        # first, sample base variable
        new_base_var_state = self.base_var_kernel._sample_from_proposal(key_base_var, state.base_var_state)

        this_iter_log_l = ThetaConditionalLogDensity(
            self.target_log_prob.log_likelihood, new_base_var_state.x
        )

        # then, sample auxiliary variable
        if self.config.aux_var_init_strategy == "x_obs":
            aux_var_init_state = state.aux_var_state.replace(x=self.target_log_prob.x_obs)
        else:
            assert self.config.aux_var_init_strategy == "warm"
            aux_var_init_state = state.aux_var_state

        from sbi_ebm.samplers.inference_algorithms.mcmc.base import _MCMCChain
        c = cast(_MCMCChain[Config_T, State_T, Info_T], state.aux_var_mcmc_chain.replace(log_prob=this_iter_log_l, _init_state=aux_var_init_state))
        new_chain, chain_res = c.run(key_aux_var)

        return state.replace(
            x=new_base_var_state.x, aux_var_state=chain_res.final_state, base_var_state=new_base_var_state, kernel_config=new_chain.config.kernel_factory.config,
            # aux_var_info=chain_res.info,
            aux_var_mcmc_chain=new_chain
        )

    def _compute_accept_prob(
        self,
        proposal: SAVMState[Config_T, State_T, Info_T],
        x: SAVMState[Config_T, State_T, Info_T],
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
                x.base_var_state.x, proposal.aux_var_state.x
            )
            - self.target_log_prob.log_likelihood(
                proposal.base_var_state.x, proposal.aux_var_state.x
            )
        )
        log_alpha = jnp.nan_to_num(log_alpha, nan=-50, neginf=-50, posinf=0)

        return log_alpha

    # def one_step(self, x: SAVMState[State_T, Info_T], key: PRNGKeyArray) -> Result[SAVMState[State_T, Info_T], SAVMInfo]:
    #     ret = cast(SAVMState[State_T, Info_T], super(SAVMKernel, self).one_step(x, key))
    #     return ret.replace(info=ret.info.replace(aux_var_info=ret.state.aux_var_info))


class SAVMKernelFactory(TunableMHKernelFactory[SAVMConfig[Config_T, State_T, Info_T], SAVMState[Config_T, State_T, Info_T], SAVMInfo[Info_T]]):
    kernel_cls: Type[SAVMKernel[Config_T, State_T, Info_T]] = struct.field(pytree_node=False, default=SAVMKernel)

    # XXX: log_prob breaks contravariance of build_kernel
    def build_kernel(self, log_prob: DoublyIntractableLogDensity) -> SAVMKernel[Config_T, State_T, Info_T]:
        return self.kernel_cls.create(log_prob, self.config)
