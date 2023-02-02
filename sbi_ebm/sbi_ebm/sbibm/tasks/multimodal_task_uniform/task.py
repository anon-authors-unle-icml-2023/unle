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


class MultiModalLikelihoodTaskUniform(Task):
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
            dim_parameters=2,
            dim_data=2,
            name="MultiModalLikelihoodTaskUniform",
            name_display="MultiModalLikelihoodTaskUniform",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )
        self.prior_params = {
            "low": -10 * torch.ones((2,)),
            "high": 10 * torch.ones((2,)),
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
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

        self._jax_prior_params = tree_map(jnp.array, self.prior_params)
        self._jax_simulator_params = tree_map(jnp.array, self.simulator_params)
        self._jax_prior_dist = convert_dist(self.prior_dist, implementation="numpyro")

        self.prior_dist.set_default_validate_args(False)

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
        from jax import random
        from sbi_ebm.samplers.smc import smc_sampler, SMCConfig, SMCParticleApproximation
        from sbi_ebm.samplers.mala import MALAConfig
        from sbi_ebm.distributions import maybe_wrap
        import jax.numpy as jnp

        key = random.PRNGKey(0)
        import numpyro.distributions as np_distributions
        assert isinstance(self._jax_prior_dist, np_distributions.Distribution)
        x0 = SMCParticleApproximation.from_npdistribution(self._jax_prior_dist, num_samples=num_samples, key=key)

        if observation is None:
            obs = jnp.array(self.get_observation(num_observation)[0])
        else:
            obs = jnp.array(observation)


        key, subkey = random.split(key)
        final_samples, _ , _ = smc_sampler(maybe_wrap(lambda x: self._jax_unnormalized_logpost(x, obs)), x0, SMCConfig(num_steps=100, mala_config=MALAConfig(num_steps=10, step_size=0.001)), subkey)

        from sbi_ebm.samplers.mala import vmapped_mala
        key, subkey = random.split(key)
        final_samples, _ , _ = vmapped_mala(maybe_wrap(lambda x: self._jax_unnormalized_logpost(x, obs)), final_samples.to_mh_particles(), MALAConfig(num_steps=200, step_size=0.001), subkey)
        return torch.Tensor(np.array(final_samples.particles.x))


if __name__ == "__main__":
    task = MultiModalLikelihoodTaskUniform()
    task._setup(1)
