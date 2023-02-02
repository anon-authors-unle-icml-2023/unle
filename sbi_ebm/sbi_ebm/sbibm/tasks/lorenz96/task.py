from pathlib import Path
from typing import Any, Callable, Optional, cast

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

from sbi_ebm.distributions import maybe_wrap
from sbi_ebm.pytypes import PRNGKeyArray
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMCParticleApproximation
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist

from sbi_ebm.sbibm.tasks.lorenz96.utils_Lorenz95_example import StochLorenz95

class Lorenz96(Task):
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
            dim_data=135,
            name="Lorenz96",
            name_display="Location Dependent Covariance Task",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
            # observation_seeds=observation_seeds,
        )

        theta1_min = 1.4
        theta1_max = 2.2
        theta2_min = 0
        theta2_max = 1

        sigma_e_min = 1.5
        sigma_e_max = 2.5
        phi_min = 0
        phi_max = 1

        self.prior_params = dict(
            min_vals=torch.Tensor([theta1_min, theta2_min, sigma_e_min, phi_min]).float(),
            max_vals=torch.Tensor([theta1_max, theta2_max, sigma_e_max, phi_max]).float()
        )

        self.prior_dist = pdist.Uniform(
            low=self.prior_params['min_vals'], high=self.prior_params['max_vals']
        ).to_event()
        self.prior_dist.set_default_validate_args(False)

        self._jax_prior_params = tree_map(jnp.array, self.prior_params)
        self._jax_prior_dist = convert_dist(self.prior_dist, implementation="numpyro")
        self._jax_prior_dist = cast(npdist.Distribution, self._jax_prior_dist)

        from abcpy.continuousmodels import Uniform

        theta1 = Uniform([[theta1_min], [theta1_max]], name='theta1')
        theta2 = Uniform([[theta2_min], [theta2_max]], name='theta2')
        sigma_e = Uniform([[sigma_e_min], [sigma_e_max]], name='sigma_e')
        phi = Uniform([[phi_min], [phi_max]], name='phi')

        self.lorenz = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=1.5,n_timestep_per_time_unit=30, K=3,name='lorenz')  # type: ignore


    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))
        return prior

    def get_prior_dist(self) -> pdist.Distribution:
        return self.prior_dist

    def get_simulator(self, max_calls=None) -> Simulator:
        def simulator(parameters):
            parameters = np.array(parameters)
            observations = np.empty((len(parameters), self.dim_data), dtype=np.float32)
            for i, param in enumerate(parameters):
                obs = self.lorenz.forward_simulate(param, 1, np.random.RandomState(i))[0]
                observations[i] = obs +  1e-10 * np.random.randn(self.dim_data).astype(np.float32)

            ret = torch.from_numpy(observations).float()
            # add some noise to ensure each observation is unique
            ret = ret + 1e-6 * torch.randn_like(ret)
            return ret
        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_transforms(
        self,
        automatic_transforms_enabled: bool = True,
        num_observation: Optional[int] = 1,
        observation: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        return {
            'parameters': pdist.transforms.biject_to(self.prior_dist.support).inv
        }

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

        key, key_init = random.split(key)
        x0 = SMCParticleApproximation.from_npdistribution(
            npdist.MultivariateNormal(
                loc=jnp.zeros((self.dim_parameters,)),
                covariance_matrix=25 * jnp.eye((self.dim_parameters)),
            ),
            num_samples=num_samples,
            key=key_init,
        )
        return torch.Tensor(np.array(x0.xs))


if __name__ == "__main__":
    task = Lorenz96()
    task._setup(n_jobs=1)
