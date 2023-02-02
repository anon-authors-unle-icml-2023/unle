from pathlib import Path
from typing import Callable, Optional

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
from sbi_ebm.samplers.mala import MALAConfig, MHParticleApproximation, vmapped_mala
from sbi_ebm.samplers.smc import SMCParticleApproximation, SMCConfig, smc_sampler
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist


class MultiModalLikelihoodTask(Task):
    def __init__(self):

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
            dim_parameters=4,
            dim_data=4,
            name="MultiModalLikelihoodTask",
            name_display="MultiModalLikelihoodTask",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        num_modes = 4
        num_dims = 4

        self._jax_prior_params = {
            "low": jnp.array([-3.0 for _ in range(self.dim_parameters)]),
            "high": jnp.array([+3.0 for _ in range(self.dim_parameters)]),
        }

        self.prior_params = tree_map(
            lambda x: torch.from_numpy(np.array(x)),
            self._jax_prior_params
        )

        mode_offsets = jnp.array([
            [1., 0, 0, 0],
            [0, 1., 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.],
        ]).astype(float)


        self._jax_simulator_params = {
            'mode_probs': jnp.ones((num_modes,)) / num_modes,
            "mode_offsets": mode_offsets,
            "covariance_matrix": 0.01 * jnp.eye(num_dims),
        }

        self.simulator_params = tree_map(
            lambda x: torch.from_numpy(np.array(x)), self._jax_simulator_params
        )

        self._jax_prior_dist = npdist.Uniform(**self._jax_prior_params).to_event()
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event()
        self.prior_dist.set_default_validate_args(False)

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_prior_dist(self):
        return self.prior_dist

    def get_observation(self, num_observation: int) -> torch.Tensor:
        return torch.zeros((self.dim_data,)).reshape(1, self.dim_data)

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

    def _log_likelihood(self, x, theta):
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

    def _unnormalized_logpost(self, x, theta):
        # XXX: does not broadcast well.
        return self._log_likelihood(x, theta) + self.get_prior_dist().log_prob(theta)

    def _jax_log_likelihood(self, x, theta):
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

    def _jax_unnormalized_logpost(self, x, theta):
        return self._jax_log_likelihood(x, theta) + self._jax_prior_dist.log_prob(theta)

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
        observation = jnp.array(observation)

        logpost = maybe_wrap(
            lambda x: self._jax_unnormalized_logpost(theta=x, x=observation)
        )

        key, key_init = random.split(key)
        x0 = SMCParticleApproximation.from_npdistribution(
            npdist.MultivariateNormal(
                loc=jnp.zeros((self.dim_parameters,)),
                covariance_matrix= jnp.eye((self.dim_parameters)),
            ),
            num_samples=num_samples,
            key=key_init,
        )
        config = SMCConfig(
            num_steps=5000, mala_config=MALAConfig(num_steps=5, step_size=0.001)
        )

        key, key_smc = random.split(key)
        posterior_samples, _, _ = smc_sampler(logpost, x0, config, key_smc)

        key, key_resampling = random.split(key)
        posterior_samples = posterior_samples.resample_and_reset_weights(key_resampling)

        # perform some MALA iterations to ensure sample diversity
        mala_x0s = posterior_samples.to_mh_particles()
        key, key_mala = random.split(key)
        mala_config = config.mala_config.replace(num_steps=1000)
        assert isinstance(mala_config, MALAConfig)
        final_sample, _, _ = vmapped_mala(
            logpost,
            mala_x0s,
            mala_config,
            key=key_mala,
        )

        return torch.Tensor(np.array(final_sample.xs))


if __name__ == "__main__":
    task = MultiModalLikelihoodTask()
    task._setup()
