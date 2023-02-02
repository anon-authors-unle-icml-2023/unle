from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as pdist
import torch
from jax import random
from jax.tree_util import tree_map
from numpyro import distributions as npdist
from sbibm.tasks import Task
from sbibm.tasks.simulator import Simulator
from torch.nn.functional import one_hot

from sbi_ebm.distributions import maybe_wrap
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCConfig
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist


class MultiModalLikelihoodTask(Task):
    def __init__(self, use_deterministic_first_observation=True):

        observation_seeds = [
            1000011,  # observation 1
            1000001,  # observation 2
            1000002,  # observation 3
            1000003,  # observation 4
            1000013,  # observation 5
            1000005,  # observation 6
            1000006,  # observation 7
            1000007,  # observation 8
            1000008,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=2,
            name="MultiModalLikelihoodTask",
            name_display="MultiModalLikelihoodTask",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )
        self.prior_params = {
            "loc": torch.zeros((2,)),
            "covariance_matrix": torch.eye(2),
        }
        self.simulator_params = {
            "mode_probs": torch.Tensor([0.25, 0.25, 0.0, 0.0]),
            "mode_offsets": 2
            * torch.stack(
                [
                    torch.Tensor([1, 1]),
                    torch.Tensor([-1, -1]),
                    torch.Tensor([1, -1]),
                    torch.Tensor([-1, 1]),
                ]
            ),
            "covariance_matrix": 0.1 * torch.eye(2),
        }
        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)

        self._jax_prior_params = tree_map(jnp.array, self.prior_params)
        self._jax_simulator_params = tree_map(jnp.array, self.simulator_params)
        self._jax_prior_dist = convert_dist(self.prior_dist, implementation="numpyro")

        self.prior_dist.set_default_validate_args(False)
        self.use_deterministic_first_observation = use_deterministic_first_observation

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_prior_dist(self):
        return self.prior_dist

    def get_simulator(self, max_calls=None) -> Callable:
        def simulator(parameters):
            num_samples = parameters.shape[0]
            mode_probs = self.simulator_params["mode_probs"]
            mode_offsets = self.simulator_params["mode_offsets"]

            mode_val = (
                pdist.Categorical(mode_probs)
                .expand_by((num_samples, 1))
                .to_event(1)
                .sample()
            )

            mean_val = (
                parameters
                + (
                    one_hot(mode_val, num_classes=len(mode_probs)).float()
                    @ mode_offsets
                )[:, 0, :]
            )

            # mean_val = mode_val * (parameters+2) + (1 - mode_val) * (parameters-2)

            S = torch.stack(
                [self.simulator_params["covariance_matrix"] for _ in range(num_samples)]
            )

            conditional = pdist.MultivariateNormal(
                loc=mean_val.unsqueeze(1), covariance_matrix=S.unsqueeze(1)
            ).expand((num_samples, 1))
            return pyro.sample("data", conditional)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _log_likelihood(self, theta, x):
        # XXX does not broadcast well

        mode_probs = self.simulator_params["mode_probs"]
        mode_offsets = self.simulator_params["mode_offsets"]
        cov_mat = self.simulator_params["covariance_matrix"]

        mean_val = theta + mode_offsets

        conditional_log_dist_per_mode = pdist.MultivariateNormal(
            loc=mean_val, covariance_matrix=cov_mat
        )

        log_probs = conditional_log_dist_per_mode.log_prob(
            x
        ) + torch.log(  # p(x|theta, mode)
            mode_probs + 1e-20
        )  # p(mode|theta)
        # sum over all modes
        return torch.logsumexp(log_probs, dim=0)

    def _unnormalized_logpost(self, theta, x):
        # XXX: does not broadcast well.
        return self._log_likelihood(theta, x) + self.get_prior_dist().log_prob(theta)

    def _jax_log_likelihood(self, theta, x):
        mode_probs = self._jax_simulator_params["mode_probs"]
        mode_offsets = self._jax_simulator_params["mode_offsets"]
        cov_mat = self._jax_simulator_params["covariance_matrix"]

        mean_val = theta + mode_offsets

        conditional_log_dist_per_mode = npdist.MultivariateNormal(
            loc=mean_val, covariance_matrix=cov_mat
        )

        log_probs = conditional_log_dist_per_mode.log_prob(
            x
        ) + jnp.log(  # p(x|theta, mode)
            mode_probs + 1e-20
        )  # p(mode|theta)
        # sum over all modes
        return jax.nn.logsumexp(log_probs, axis=0)

    def _jax_unnormalized_logpost(self, theta, x):
        return self._jax_log_likelihood(theta, x) + self._jax_prior_dist.log_prob(theta)

    def _sample_reference_posterior(
        self,
        num_observation: int,
        num_samples: int,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: z-score the posterior before sampling by sampling from the
        # prior-simulator joint distribution to obtain empirical mean and stds.
        key = random.PRNGKey(num_observation)

        if observation is None:
            observation = self.get_observation(num_observation)[0]
        assert observation is not None
        assert len(observation.shape) == 1
        observation = jnp.array(observation)  # type: ignore

        logpost = maybe_wrap(
            lambda x: self._jax_unnormalized_logpost(theta=x, x=observation)
        )

        init_dist = npdist.MultivariateNormal(
            loc=jnp.zeros((self.dim_parameters,)),
            covariance_matrix=25 * jnp.eye((self.dim_parameters)),
        )
        config = SMCConfig(
            num_samples=num_samples,
            num_steps=100, ess_threshold=0.8,
            inner_kernel_factory=MALAKernelFactory(MALAConfig(step_size=0.001)),
            inner_kernel_steps=5,
            record_trajectory=False
        )
        smc = SMC(config=config, log_prob=logpost)

        key, key_init = random.split(key)
        smc = smc.init(key_init, init_dist)

        key, key_smc = random.split(key)
        smc, results = smc.run(key_smc)
        return torch.Tensor(np.array(results.samples.xs))

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            # fix observation 0 to be at 0 for the paper's illustrative example
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            if self.use_deterministic_first_observation and num_observation == 1:
                observation = torch.zeros((1, self.dim_data))
                true_parameters = - 2 * torch.ones((1, self.dim_parameters))
            else:
                true_parameters = prior(num_samples=1)
                simulator = self.get_simulator()
                observation = simulator(true_parameters)

            self._save_true_parameters(num_observation, true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,  # type: ignore
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        Parallel(n_jobs=n_jobs, verbose=50, backend="sequential")(
            delayed(run)(num_observation, observation_seed, **kwargs)
            for num_observation, observation_seed in enumerate(
                self.observation_seeds, start=1
            )
        )



if __name__ == "__main__":
    task = MultiModalLikelihoodTask()
    task._setup()
