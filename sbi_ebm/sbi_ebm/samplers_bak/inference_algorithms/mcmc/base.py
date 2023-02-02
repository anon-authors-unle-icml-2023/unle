from typing import Generic, Optional, Tuple, cast
from typing_extensions import Self, Type

import jax
import jax.numpy as jnp
import numpyro.distributions as np_distributions
from flax import struct
from jax import random, vmap
from jax.lax import scan  # type: ignore
from jax.tree_util import tree_map

from sbi_ebm.pytypes import Array, LogDensity_T, PRNGKeyArray
from sbi_ebm.samplers.inference_algorithms.base import (
    InferenceAlgorithm, InferenceAlgorithmConfig, InferenceAlgorithmFactory, InferenceAlgorithmInfo,
    InferenceAlgorithmResults)
from sbi_ebm.samplers.kernels.adaptive_mala import AdaptiveMALAState
from sbi_ebm.samplers.kernels.hmc import HMCInfo, HMCKernel, HMCKernelFactory
from sbi_ebm.samplers.kernels.nuts import NUTSInfo, NUTSKernelFactory
from sbi_ebm.samplers.kernels.savm import SAVMState

from ...kernels.base import (Array_T, Config_T, Config_T_co, Info_T, Info_T,
                             Kernel, KernelFactory, MHKernelFactory, Result, State, State_T, TunableKernel, TunableMHKernelFactory)
from ...particle_aproximation import ParticleApproximation


class _MCMCChainConfig(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_steps: int = struct.field(pytree_node=False)
    record_trajectory: bool = struct.field(pytree_node=False)
    num_warmup_steps: int = struct.field(pytree_node=False)
    adapt_step_size: bool = struct.field(pytree_node=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False)



class _SingleChainResults(Generic[State_T, Info_T], struct.PyTreeNode):
    final_state: State_T
    chain: State_T
    info: Info_T
    warmup_info: Optional[Info_T] = None


class _MCMCChain(Generic[Config_T, State_T, Info_T], struct.PyTreeNode):
    config: _MCMCChainConfig[Config_T, State_T, Info_T]
    log_prob: LogDensity_T
    _init_state: Optional[State_T] = None

    def init(self, x0: Array) -> Self:
        init_state = self.config.kernel_factory.build_kernel(self.log_prob).init_state(x0)
        return self.replace(_init_state=init_state)

    def run(self, key: PRNGKeyArray) -> Tuple[Self, _SingleChainResults[State_T, Info_T]]:
        if self.config.num_warmup_steps > 0:
            key, subkey = random.split(key)
            self, warmup_info = self.warmup(subkey)
        else:
            warmup_info = None

        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        def step_fn(
            x: State_T, key: PRNGKeyArray
        ) -> Tuple[State_T, Optional[Result[State_T, Info_T]]]:
            mala_result = kernel.one_step(x, key)
            # tuple of carry, scan-concatenated vals.
            if not self.config.record_trajectory:
                output = None
            else:
                output = mala_result
            return mala_result.state, output

        assert self._init_state is not None
        init_state = self._init_state

        keys = random.split(key, self.config.num_steps)
        final_state, outputs = scan(step_fn, init_state, keys)
        if self.config.record_trajectory:
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)
        return self, _SingleChainResults(final_state, chain, stats, warmup_info)


    def warmup(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        if False and isinstance(self.config.kernel_factory, HMCKernelFactory):
            return self._warmup_blackjax(key)
        elif False and isinstance(self.config.kernel_factory, NUTSKernelFactory):
            return self._warmup_blackjax(key)
        else:
            return self._warmup_sbi_ebm(key)

    def _warmup_blackjax(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        assert isinstance(self.config.kernel_factory, TunableMHKernelFactory)

        kernel = self.config.kernel_factory.build_kernel(self.log_prob)
        import blackjax
        assert self._init_state is not None
        init_state = self._init_state

        # print(len(self.config.kernel_factory.config.inverse_mass_matrix.shape))
        if isinstance(self.config.kernel_factory, HMCKernelFactory):
            warmup = blackjax.window_adaptation(
                algorithm=blackjax.hmc,  # type: ignore
                logprob_fn=self.log_prob,
                is_mass_matrix_diagonal=len(self.config.kernel_factory.config.inverse_mass_matrix.shape) == 1,
                num_steps=self.config.num_warmup_steps,
                target_acceptance_rate=0.8,
                num_integration_steps=self.config.kernel_factory.config.num_integration_steps,
            )

        elif isinstance(self.config.kernel_factory, NUTSKernelFactory):
            warmup = blackjax.window_adaptation(
                algorithm=blackjax.nuts,  # type: ignore
                logprob_fn=self.log_prob,
                is_mass_matrix_diagonal=len(self.config.kernel_factory.config.inverse_mass_matrix.shape) == 1,
                num_steps=self.config.num_warmup_steps,
                target_acceptance_rate=0.8,
            )
        else:
            raise ValueError

        state, final_kernel, blah = warmup.run(key, init_state._blackjax_state.position)  # type: ignore
        was = blah[2]
        from blackjax.adaptation.window_adaptation import WindowAdaptationState
        assert isinstance(was, WindowAdaptationState)

        final_step_size = jnp.exp(was.da_state.log_step_size_avg[-1])
        final_imm = was.mm_state.inverse_mass_matrix[-1]

        new_kernel_factory = self.config.kernel_factory.replace(
            config=self.config.kernel_factory.config.replace(
                step_size=final_step_size, inverse_mass_matrix=final_imm
            )
        )

        new_init_state = new_kernel_factory.build_kernel(self.log_prob).init_state(state.position)
        new_chain = self.replace(config=self.config.replace(kernel_factory=new_kernel_factory), _init_state=new_init_state)
        stats = blah[1]

        accept = stats.is_accepted if isinstance(self.config.kernel_factory, HMCKernelFactory) else jnp.log(stats.acceptance_probability)

        stats = kernel._build_info(accept=accept, log_alpha=jnp.log(stats.acceptance_probability))
        return new_chain, stats

    def _warmup_sbi_ebm(self, key: PRNGKeyArray) -> Tuple[Self, Info_T]:
        if self.config.adapt_mass_matrix or self.config.adapt_step_size:
            assert isinstance(self.config.kernel_factory, TunableMHKernelFactory)
        kernel = self.config.kernel_factory.build_kernel(self.log_prob)

        # record_trajectory = self.config.record_trajectory
        record_trajectory = False

        def step_fn(carry: Tuple[State_T, AdaptiveMALAState], key: PRNGKeyArray) -> Tuple[Tuple[State_T, AdaptiveMALAState], Optional[Result[State_T, Info_T]]]:
            this_kernel = kernel
            x, adaptation_state = carry
            if self.config.adapt_step_size:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_step_size(adaptation_state.sigma)

            if self.config.adapt_mass_matrix:
                assert isinstance(this_kernel, TunableKernel)
                this_kernel = this_kernel.set_inverse_mass_matrix(adaptation_state.C)

            mala_result = this_kernel.one_step(x, key)

            next_adaptation_state = adaptation_state

            if self.config.adapt_step_size:
                next_adaptation_state = next_adaptation_state.update_sigma(log_alpha=getattr(mala_result.info, "log_alpha", 0), gamma_n=1/(next_adaptation_state.iter_no+1)**0.5)
            if self.config.adapt_mass_matrix:
                next_adaptation_state = next_adaptation_state.update_cov(x=mala_result.state.x, gamma_n=1/(next_adaptation_state.iter_no+1)**0.5)

            next_adaptation_state = next_adaptation_state.replace(iter_no=next_adaptation_state.iter_no + 1, x=mala_result.state.x)

            # tuple of carry, scan-concatenated vals.
            # if not self.config.record_trajectory:
            if not record_trajectory:
                output = None
            else:
                output = mala_result
            return (mala_result.state, next_adaptation_state), output

        assert self._init_state is not None
        init_state = self._init_state

        target_accept_rate = 0.2
        if self.config.adapt_mass_matrix:
            assert isinstance(kernel, TunableKernel)
            uses_diagonal_mass_matrix = len(kernel.get_inverse_mass_matrix().shape) == 1
            if uses_diagonal_mass_matrix:
                init_adaptation_state = AdaptiveMALAState(init_state.x, 1, init_state.x, jnp.zeros((init_state.x.shape[0],)), 1., target_accept_rate)
            else:
                init_adaptation_state = AdaptiveMALAState(init_state.x, 1, init_state.x, jnp.zeros((init_state.x.shape[0], init_state.x.shape[0])), 1., target_accept_rate)
        else:
            init_adaptation_state = AdaptiveMALAState(init_state.x, 1, None, None, 1., target_accept_rate)

        keys = random.split(key, self.config.num_warmup_steps)
        final_state, outputs = scan(step_fn, (init_state, init_adaptation_state), keys)
        if record_trajectory:
            assert outputs is not None
            stats, chain = outputs.info, outputs.state
        else:
            _smoke_res = kernel.one_step(init_state, key)
            stats = _smoke_res.info
            chain = tree_map(lambda x: x[None, ...], final_state)


        # new_init_state = self.config.kernel_factory.build_kernel(self.log_prob).init_state(final_state[0].x)
        new_init_state = final_state[0]


        final_kernel = kernel

        if self.config.adapt_step_size:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_step_size(step_size=final_state[1].sigma)
        if self.config.adapt_mass_matrix:
            assert isinstance(final_kernel, TunableKernel)
            final_kernel = final_kernel.set_inverse_mass_matrix(final_state[1].C)

        return self.replace(
            config=self.config.replace(kernel_factory=self.config.kernel_factory.replace(config=final_kernel.config)),
            _init_state=new_init_state
        ), stats


class MCMCConfig(
    Generic[Config_T, State_T, Info_T], InferenceAlgorithmConfig, struct.PyTreeNode
):
    kernel_factory: KernelFactory[Config_T, State_T, Info_T]
    num_samples: int = struct.field(pytree_node=False)
    num_chains: int = struct.field(pytree_node=False, default=100)
    thinning_factor: int = struct.field(pytree_node=False, default=10)
    record_trajectory: bool = struct.field(pytree_node=False, default=True)
    num_warmup_steps: int = struct.field(pytree_node=False, default=0)
    adapt_step_size: bool = struct.field(pytree_node=False, default=False)
    adapt_mass_matrix: bool = struct.field(pytree_node=False, default=False)
    resample_stuck_chain_at_warmup: bool = struct.field(pytree_node=False, default=False)


class MCMCInfo(Generic[State_T, Info_T], InferenceAlgorithmInfo):
    single_chain_results: _SingleChainResults[State_T, Info_T]


class MCMCResults(Generic[State_T, Info_T], InferenceAlgorithmResults):
    samples: ParticleApproximation
    info: MCMCInfo[State_T, Info_T]


class MCMCAlgorithm(
    InferenceAlgorithm[MCMCConfig[Config_T, State_T, Info_T]]
):
    _single_chains: Optional[_MCMCChain[Config_T, State_T, Info_T]] = None

    @classmethod
    def create(cls, config: MCMCConfig[Config_T, State_T, Info_T], log_prob: LogDensity_T) -> Self:
        # build single chain MCMC configs
        num_total_steps = (config.num_samples * config.thinning_factor) / config.num_chains
        assert num_total_steps == int(num_total_steps)
        _single_chain_configs = vmap(lambda _: _MCMCChainConfig(config.kernel_factory, int(num_total_steps), True, config.num_warmup_steps, config.adapt_step_size, config.adapt_mass_matrix))(jnp.arange(config.num_chains))

        # _single_chains = vmap(lambda c: _MCMCChain(c, log_prob), in_axes=_MCMCChain(0, None, 0))(_single_chain_configs)
        # _single_chains = vmap(_MCMCChain, in_axes=(0, None))(_single_chain_configs, log_prob)
        _single_chains = vmap(_MCMCChain, in_axes=(0, None), out_axes=_MCMCChain(0, None, None))(_single_chain_configs, log_prob)  # type: ignore
        return cls(config, log_prob, _init_state=None, _single_chains=_single_chains)

    def init(self, key: PRNGKeyArray, dist: np_distributions.Distribution, reweight_and_resample: bool = False) -> Self:
        xs = dist.sample(key, (self.config.num_chains,))
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_chains))
        if reweight_and_resample:
            key, subkey = random.split(key)
            log_ratio = vmap(self.log_prob)(init_state.xs) - vmap(dist.log_prob)(init_state.xs)
            init_state = init_state.replace(log_ws=log_ratio).resample_and_reset_weights(subkey)

        # single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0))(self._single_chains, init_state.particles)
        single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0), in_axes=(_MCMCChain(0, None, None), 0), out_axes=_MCMCChain(0, None, 0))(self._single_chains, init_state.particles)  # type: ignore

        return self.replace(_init_state=init_state, _single_chains=single_chains)

    def init_from_particles(self, xs: Array) -> Self:
        assert len(xs.shape) == 2
        assert len(xs) == self.config.num_chains
        init_state = ParticleApproximation(xs, jnp.zeros(self.config.num_samples))

        # single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0))(self._single_chains, init_state.particles)
        single_chains = vmap(lambda c, x0: cast(_MCMCChain[Config_T, State_T, Info_T], c).init(x0), in_axes=(_MCMCChain(0, None, None), 0), out_axes=_MCMCChain(0, None, 0))(self._single_chains, init_state.particles)  # type: ignore

        return self.replace(_init_state=init_state, _single_chains=single_chains)

    def set_log_prob(self, log_prob: LogDensity_T) -> Self:
        self = self.replace(log_prob=log_prob)
        if self._single_chains is not None:
            self = self.replace(_single_chains=self._single_chains.replace(log_prob=log_prob))
        return self

    def set_num_warmup_steps(self, num_warmup_steps) -> Self:
        self = cast(Self, self.replace(config=self.config.replace(num_warmup_steps=num_warmup_steps)))
        if self._single_chains is not None:
            self = self.replace(_single_chains=self._single_chains.config.replace(num_warmup_steps=num_warmup_steps))
        return self

    def _maybe_prewarmup_and_resample(self, key: PRNGKeyArray) -> _MCMCChain[Config_T, State_T, Info_T]:
        assert self._single_chains is not None
        new_single_chains = self._single_chains
        if self.config.resample_stuck_chain_at_warmup and self.config.num_warmup_steps > 0 and isinstance(self.config.kernel_factory, MHKernelFactory):
            key, subkey = random.split(key)
            keys = random.split(subkey, self.config.num_chains)
            _num_total_steps = new_single_chains.config.num_steps
            new_single_chains, single_chain_results = vmap(
                lambda c, k: cast(_MCMCChain[Config_T, State_T, Info_T], c).replace(config=c.config.replace(num_warmup_steps=max(self.config.num_warmup_steps // 10, 1), num_steps=1)).run(k),
                # in_axes=(0, 0), out_axes=(0, 0)
                in_axes=(_MCMCChain(0, None, 0), 0), out_axes=(_MCMCChain(0, None, 0), 0)  # type: ignore
            )(new_single_chains, keys)
            new_single_chains = new_single_chains.replace(config=new_single_chains.config.replace(num_warmup_steps=self.config.num_warmup_steps, num_steps=_num_total_steps))

            key, subkey = random.split(key)
            assert single_chain_results.warmup_info is not None
            accept_rate_per_chain = jnp.mean(single_chain_results.warmup_info.accept, axis=-1)
            _probs = jnp.minimum(0.01, accept_rate_per_chain)
            _probs /= jnp.sum(_probs)
            indices = np_distributions.Categorical(probs=_probs).sample(subkey, sample_shape=(self.config.num_chains,))
            new_single_chains = cast(_MCMCChain[Config_T, State_T, Info_T], jax.lax.cond(jnp.min(accept_rate_per_chain) < 0.01, lambda: tree_map(lambda x: jnp.take(x, indices, axis=0), new_single_chains), lambda: new_single_chains))
        return new_single_chains


    def run(self, key: PRNGKeyArray) -> Tuple[Self, MCMCResults[State_T, Info_T]]:
        key, subkey = random.split(key)
        new_single_chains = self._maybe_prewarmup_and_resample(subkey)

        key, subkey = random.split(key)
        new_single_chains, single_chain_results = vmap(
            lambda c, k: cast(_MCMCChain[Config_T, State_T, Info_T], c).run(k),
            # in_axes=(0, 0), out_axes=(0, 0)
            in_axes=(_MCMCChain(0, None, 0), 0), out_axes=(_MCMCChain(0, None, 0), 0)  # type: ignore
        )(new_single_chains, random.split(subkey, self.config.num_chains))

        final_samples = single_chain_results.chain.x[:, ::-self.config.thinning_factor, :].reshape(-1, single_chain_results.chain.x.shape[-1])
        assert len(final_samples) == self.config.num_samples

        new_self = self.replace(_single_chains=new_single_chains)
        return new_self, MCMCResults(
                ParticleApproximation(final_samples, jnp.zeros((final_samples.shape[0],))),
                info=MCMCInfo(single_chain_results)
            )



class MCMCAlgorithmFactory(InferenceAlgorithmFactory[MCMCConfig[Config_T, State_T, Info_T]]):
    def build_algorithm(self, log_prob: LogDensity_T) -> MCMCAlgorithm[Config_T, State_T, Info_T]:
        return MCMCAlgorithm.create(log_prob=log_prob, config=self.config)
