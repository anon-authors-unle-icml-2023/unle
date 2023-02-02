from pathlib import Path
from typing import Callable, Optional

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
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCConfig
from sbi_ebm.samplers.kernels.mala import MALAConfig, MALAKernelFactory
from sbi_ebm.sbibm.pyro_to_numpyro import convert_dist


def _make_rot_mat(thet):
    ret = np.stack(
        (
            np.array([np.cos(thet), np.sin(thet)]),
            np.array([-np.sin(thet), np.cos(thet)]),
        ),
        axis=1,
    )
    return torch.from_numpy(ret)


def _jax_make_rot_mat(thet):
    ret = jnp.stack(
        (
            jnp.array([jnp.cos(thet), jnp.sin(thet)]),
            jnp.array([-jnp.sin(thet), jnp.cos(thet)]),
        ),
        axis=1,
    )
    return ret


class LDCT(Task):
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
            name="LDCT",
            name_display="Location Dependent Covariance Task",
            num_observations=10,
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
            # observation_seeds=observation_seeds,
        )
        self.prior_params = {
            "loc": torch.zeros((2,)),
            "covariance_matrix": 25 * torch.eye(2),
        }
        self.simulator_params = {
            "variance_vecs": torch.Tensor([16, 0.25]),
        }

        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

        self._jax_prior_params = tree_map(jnp.array, self.prior_params)
        self._jax_simulator_params = tree_map(jnp.array, self.simulator_params)
        self._jax_prior_dist = convert_dist(self.prior_dist, implementation="numpyro")

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_prior_dist(self) -> pdist.Distribution:
        return self.prior_dist

    def get_simulator(self, max_calls=None) -> Callable:
        def simulator(parameters):
            num_samples = parameters.shape[0]
            cov_mats = torch.empty(num_samples, 2, 2)
            vars_vec = self.simulator_params["variance_vecs"]

            for i, param in enumerate(parameters):
                rot_mat = _make_rot_mat(0.5 * param[0])
                cov_mats[i] = rot_mat @ torch.diag(vars_vec) @ rot_mat.T

            conditional = pdist.MultivariateNormal(
                loc=parameters, covariance_matrix=cov_mats
            )
            return pyro.sample("data", conditional)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _log_likelihood(self, theta, x):
        # XXX does not broadcast well
        rot_mat = _make_rot_mat(0.5 * theta[0])
        vars_vec = self.simulator_params["variance_vecs"]
        cov_mat = rot_mat @ torch.diag(vars_vec) @ rot_mat.T

        conditional = pdist.MultivariateNormal(loc=theta, covariance_matrix=cov_mat)
        return conditional.log_prob(x)

    def _unnormalized_logpost(self, theta, x):
        # XXX: does not broadcast well.
        return self._log_likelihood(theta, x) + self.get_prior_dist().log_prob(theta)

    def _jax_log_likelihood(self, theta, x):
        # XXX does not broadcast well
        rot_mat = jnp.array(_jax_make_rot_mat(0.5 * theta[0]))
        vars_vec = self._jax_simulator_params["variance_vecs"]
        cov_mat = rot_mat @ jnp.diag(vars_vec) @ rot_mat.T  # type: ignore

        conditional = npdist.MultivariateNormal(loc=theta, covariance_matrix=cov_mat)
        return conditional.log_prob(x)

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


if __name__ == "__main__":
    task = LDCT()
    task._setup()
