from typing import Any, Type
from typing import Callable, Optional, Tuple, Union
from typing_extensions import Self

from flax import struct
from jax import random, vmap
from jax.nn import logsumexp, softmax
import jax.numpy as jnp
from numpyro import distributions as np_distributions
from sbi_ebm.distributions import DoublyIntractableJointLogDensity, ThetaConditionalLogDensity

from sbi_ebm.pytypes import Array, LogLikelihood_T, Numeric, PRNGKeyArray
from sbi_ebm.samplers.kernels.base import Info, KernelConfig, MHKernel, MHKernelFactory, State
from sbi_ebm.samplers.kernels.discrete_gibbs import DiscreteLogDensity
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

class ThetaConditionalDist(np_distributions.Distribution):
    arg_constraints = {"likelihood_factory": None, "prior": None, "x": None}
    meta_fields = {"z_transform": None, "x_transform": None}

    def __init__(
        self,
        log_likelihood: LogLikelihood_T,
        x_0: Array,
    ):
        self.log_likelihood = log_likelihood
        super(ThetaConditionalDist, self).__init__(
            batch_shape=(), event_shape=x_0.shape
        )

    def sample(self, key, sample_shape=...):
        return jnp.zeros(sample_shape)
    


class MixedJointLogDensity(DoublyIntractableJointLogDensity):
    """
    Discrete relaxation of a unnormalized log density of the form:

        p(x) = exp(-E(x)) g(x)


    Where g is a base measure with available IID samples x_i which are used to
    approximate p using:

        p(x) = \sum_i exp(-E(x_i)) \delta(x_i)
    """
    thetas: Array = struct.field(pytree_node=False, default=None)


    def get_discrete_given_continuous(self, x: Array):
        log_probs = vmap(self.log_likelihood, in_axes=(0, None))(self.thetas, x)
        from jax.nn import softmax
        probs = softmax(log_probs)
        return np_distributions.Categorical(probs=probs)


    def get_continuous_given_discrete(self, theta: Array):
        return ThetaConditionalLogDensity(self.log_likelihood, theta)



class MixedHMCConfig(KernelConfig):
    step_size: float
    C: Optional[Array_T] = None
    max_tree_depth: int = struct.field(pytree_node=False, default=10)


class MixedHMCInfo(Info):
    accept: Numeric
    log_alpha: Numeric
    accept_prob: Numeric


class MixedHMCState(State):
    _numpyro_state: np_HMCGibbstate

class MixedHMCKernel(TunableKernel[MixedHMCConfig, MixedHMCState, MixedHMCInfo]):
    supports_diagonal_mass: bool = True
    model: Any = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls: Type[Self], target_log_prob: LogDensity_T, config: MixedHMCConfig
    ) -> Self:

        assert isinstance(target_log_prob, MixedJointLogDensity)
        def model():
            import numpyro
            theta_idx = numpyro.sample("theta_idx", np_distributions.Categorical(probs=jnp.ones(target_log_prob.thetas.shape[0])))
            numpyro.sample("x", ThetaConditionalDist(target_log_prob.log_likelihood, target_log_prob.thetas[theta_idx]))

        return cls(target_log_prob, config, model=model)

    def _sample_from_proposal(self, key: PRNGKeyArray, x: MixedHMCState) -> MixedHMCState:
        raise NotImplementedError

    def _build_info(self, accept: Numeric, log_alpha: Numeric) -> MixedHMCInfo:
        return MixedHMCInfo(accept=accept, log_alpha=log_alpha, accept_prob=jnp.exp(log_alpha))

    def _compute_accept_prob(self, proposal: MixedHMCState, x: MixedHMCState) -> Numeric:
        raise NotImplementedError

    def get_step_size(self) -> Array_T:
        return self.config.step_size

    def get_inverse_mass_matrix(self) -> Array_T:
        return self.config.C

    def set_step_size(self, step_size) -> Self:
        return self.replace(config=self.config.replace(step_size=step_size))

    def _set_dense_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def _set_diag_inverse_mass_matrix(self, inverse_mass_matrix: Array_T) -> Self:
        return self.replace(config=self.config.replace(C=inverse_mass_matrix))

    def init_state(self, x: Array_T):
        assert isinstance(self.target_log_prob, MixedJointLogDensity)
        self.target_log_prob
        assert self.model is not None

        def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
            assert isinstance(self.target_log_prob, MixedJointLogDensity)
            new_theta = self.target_log_prob.get_discrete_given_continuous(hmc_sites["x"])
            return {"theta_idx": new_theta.sample(rng_key)}

        hmc_kernel = NUTS(self.model)
        hmcgibbs_kernel = HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=["theta_idx"])
        from numpyro.infer import MCMC
        mcmc = MCMC(hmcgibbs_kernel, num_warmup=100, num_samples=100, progress_bar=False)
        return mcmc.run(random.PRNGKey(0), x)

class MixedHMCKernelFactory(TunableMHKernelFactory[MixedHMCConfig, MixedHMCState, MixedHMCInfo]):
    kernel_cls: Type[MixedHMCKernel] = struct.field(pytree_node=False, default=MixedHMCKernel)

    def build_kernel(self, log_prob: LogDensity_T) -> MixedHMCKernel:
        return self.kernel_cls.create(log_prob, self.config)
