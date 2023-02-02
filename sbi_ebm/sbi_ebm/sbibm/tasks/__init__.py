from typing import Type
from typing_extensions import Self
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pyro_distributions
import sbibm
import torch
from jax import random
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from sbibm.tasks import Task
from sbi_ebm.distributions import maybe_wrap

from sbi_ebm.pytypes import Array, LogDensity_T, Numeric, Simulator_T
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCConfig
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist
from sbi_ebm.sbibm.tasks.lorenz96.task import Lorenz96
from sbi_ebm.sbibm.tasks.ornstein_uhlenbeck.task import OrnsteinUhlenbeck

from .ldct.task import LDCT
from .multimodal_task.task import MultiModalLikelihoodTask


def get_task(task_name: str) -> Task:
    if task_name == "LDCT":
        return LDCT()
    elif task_name == "Lorenz96":
        return Lorenz96()
    elif task_name == "ornstein_uhlenbeck":
        return OrnsteinUhlenbeck()
    elif task_name == "MultiModalLikelihoodTask":
        return MultiModalLikelihoodTask()
    elif task_name == "MultiModalLikelihoodTaskUniform":
        from sbi_ebm.sbibm.tasks.multimodal_task_uniform.task import MultiModalLikelihoodTaskUniform  # type: ignore
        return MultiModalLikelihoodTaskUniform()
    # elif task_name == "pyloric":
    #     from sbi_ebm.sbibm.tasks.pyloric.task import Pyloric
    #     return Pyloric()
    elif task_name == "pyloric":
        from sbi_ebm.sbibm.tasks.pyloric_stg import Pyloric  # type: ignore
        return Pyloric()
    else:
        return sbibm.get_task(task_name)


class JaxTask:
    def __init__(self, task: Task) -> None:
        self.task = task

    def get_prior_dist(self) -> np_distributions.Distribution:
        prior_dist = self.task.get_prior_dist()
        assert isinstance(prior_dist, pyro_distributions.Distribution), prior_dist

        if self.task.name == "Lorenz96":
            p = getattr(self.task, "_jax_prior_dist", None)
            assert isinstance(p, np_distributions.Distribution)
            return p

        converted_prior_dist = convert_dist(prior_dist, implementation="numpyro")
        assert isinstance(converted_prior_dist, np_distributions.Distribution)

        # if self.task.name == "slcp":
        #     converted_prior_dist = np_distributions.MultivariateNormal(
        #         jnp.zeros((self.task.dim_parameters,)),
        #         jnp.eye(self.task.dim_parameters),
        #     )

        return converted_prior_dist

    @classmethod
    def from_task_name(cls: Type[Self], task_name: str) -> Self:
        from sbi_ebm.sbibm.tasks import get_task
        return cls(get_task(task_name))

    def get_simulator(self) -> Simulator_T:
        pyro_simulator = self.task.get_simulator()

        def simulator(thetas: Array) -> Array:
            torch_thetas = torch.from_numpy(np.array(thetas)).float()
            from sbi_ebm.dtypes import should_use_float64
            if should_use_float64():
                return jnp.array(pyro_simulator(torch_thetas), dtype=jnp.float64)
            else:
                return jnp.array(pyro_simulator(torch_thetas))
        return simulator

    def get_observation(self, num_observation: int) -> Array:
        return jnp.array(self.task.get_observation(num_observation))

    def _parameter_event_space_bijector(self) -> np_transforms.Transform:
        prior_dist = self.get_prior_dist()
        return np_distributions.biject_to(prior_dist.support)

    def __reduce__(self):
        return JaxTask.from_task_name, (self.task.name,)


class ReferencePosterior(np_distributions.Distribution):
    arg_contraints = {"log_prob": None}

    def __init__(self, task: JaxTask, log_prob: LogDensity_T, num_observation: int):
        self._task = task
        self._log_prob = log_prob
        self._num_observation = num_observation
        self.support = self._task.get_prior_dist().support

        self._smc_config = SMCConfig(
            num_samples=1000,
            num_steps=5000, ess_threshold=0.8,
            inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
            inner_kernel_steps=5,
            record_trajectory=False
        )

        super(ReferencePosterior, self).__init__(
            batch_shape=(), event_shape=(self._task.task.dim_parameters,)
        )

    def log_prob(self, z: Array) -> Numeric:
        return self._log_prob(z)

    def sample(self, key, sample_shape=()) -> Array:
        log_prob = lambda x: self.log_prob(x) +  1e-15 * self._task.get_prior_dist().log_prob(x)

        key, key_init = random.split(key)
        smc = SMC(config=self._smc_config.replace(num_samples=sample_shape[0]), log_prob=maybe_wrap(log_prob))
        smc = smc.init(key=key_init, dist=self._task.get_prior_dist())

        key, key_sampling = random.split(key)
        smc, results = smc.run(key_sampling)

        key, key_resampling = random.split(key)
        samples = results.samples.resample_and_reset_weights(key_resampling)


        # perform some MALA iterations to ensure sample diversity
        # mala_x0s = samples.to_mh_particles()
        # key, key_mala = random.split(key)

        # samples, _, _ = vmapped_mala(
        #     maybe_wrap(self.log_prob),
        #     mala_x0s,
        #     self._diversity_config,
        #     key=key_mala,
        # )

        # samples = inv_transform(samples.xs)
        samples = samples.xs

        samples = jnp.maximum(
            samples, self._task.get_prior_dist().support.base_constraint.lower_bound + 1e-5  # type: ignore
        )
        samples = jnp.minimum(
            samples, self._task.get_prior_dist().support.base_constraint.upper_bound - 1e-5  # type: ignore
        )

        print(jnp.min(samples))
        print(jnp.max(samples))

        return samples


from pyro import distributions as pdist
from pyro.distributions import MultivariateNormal

class TorchReferencePosterior(pdist.Distribution):
    def __init__(self, jax_reference_posterior: ReferencePosterior, use_reference_samples=True, prior_prop=0.):
        self._jax_reference_posterior = jax_reference_posterior
        super(TorchReferencePosterior, self).__init__()
        self._dummy_sample = False
        self._use_reference_samples = use_reference_samples
        self.support = self._jax_reference_posterior._task.task.get_prior_dist().support
        self.prior_prop = prior_prop


    def log_prob(self, x, *args, **kwargs):
        if x.shape[-1] != self._jax_reference_posterior._task.task.dim_parameters:
            raise ValueError("x.shape[-1] != self._jax_reference_posterior._task.task.dim_parameters")
        return 0 * torch.ones(x.shape[:-1])

    def sample(self, sample_shape=()):
        if self._dummy_sample:
            event_shape = self._jax_reference_posterior.event_shape
            sample_shape = sample_shape + event_shape
            return torch.zeros(*sample_shape)
        elif self._use_reference_samples:
            posterior_samples = self._jax_reference_posterior._task.task.get_reference_posterior_samples(
                self._jax_reference_posterior._num_observation
            )
            # sample from the task prior
            prior_samples = self._jax_reference_posterior._task.task.get_prior_dist().sample(sample_shape)

            if len(sample_shape) > 0:
                samples_idx = torch.randint(0, posterior_samples.shape[0], (sample_shape[0],))
                posterior_samples = posterior_samples[samples_idx]
            else:
                samples_idx = torch.randint(0, posterior_samples.shape[0], (1,))
                posterior_samples =  posterior_samples[samples_idx[0], :]

            print(posterior_samples.shape, prior_samples.shape)
            
            should_use_prior = (pdist.Uniform(0, 1).sample((*sample_shape, 1)) < self.prior_prop).float()  # type: ignore

            return prior_samples * should_use_prior + posterior_samples * (1 - should_use_prior)

        else:
            import random as rnd
            random_seed = rnd.randint(0, 2**32)
            jax_key = random.PRNGKey(random_seed)
            return torch.from_numpy(np.array(self._jax_reference_posterior.sample(jax_key, sample_shape)))







def get_reference_posterior(task: JaxTask, num_observation: int) -> ReferencePosterior:
    x_obs = jnp.array(task.task.get_observation(num_observation)[0])

    if task.task.name != "slcp":
        raise ValueError
    else:

        def log_prob(z):
            m = z[:2]
            s1 = z[2] ** 2
            s2 = z[3] ** 2
            rho = jnp.tanh(z[4])

            eps = 0.000001
            S = jnp.array(
                [[eps + s1 ** 2, rho * s1 * s2], [rho * s1 * s2, eps + s2 ** 2]]
            )
            det_S = (eps + s1 ** 2) * (eps + s2 ** 2) - (rho * s1 * s2) ** 2
            S_inv = (
                1
                / det_S
                * jnp.array(
                    [[eps + s2 ** 2, -rho * s1 * s2], [-rho * s1 * s2, eps + s1 ** 2]]
                )
            )

            return (
                -0.5 * jnp.log(det_S)
                - 0.5 * (x_obs[:2] - m) @ S_inv @ (x_obs[:2] - m)  # type: ignore
                + -0.5 * jnp.log(det_S)
                - 0.5 * (x_obs[2:4] - m) @ S_inv @ (x_obs[2:4] - m)  # type: ignore
                + -0.5 * jnp.log(det_S)
                - 0.5 * (x_obs[4:6] - m) @ S_inv @ (x_obs[4:6] - m)  # type: ignore
                + -0.5 * jnp.log(det_S)
                - 0.5 * (x_obs[6:] - m) @ S_inv @ (x_obs[6:] - m)  # type: ignore
                + task.get_prior_dist().log_prob(z)
            )
            # dist = np_distributions.MultivariateNormal(m, covariance_matrix=S)
            # return (
            #     dist.log_prob(x_obs[:2])
            #     + dist.log_prob(x_obs[2:4])
            #     + dist.log_prob(x_obs[4:6])
            #     + dist.log_prob(x_obs[6:])
            #     + task.get_prior_dist().log_prob(z)
            # )

        return ReferencePosterior(task, log_prob, num_observation)  # type: ignore
