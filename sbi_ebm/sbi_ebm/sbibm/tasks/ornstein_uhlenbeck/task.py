from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import pyro
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task

from pyro.distributions.torch import TransformedDistribution  # type: ignore
from pyro.distributions.transforms.lower_cholesky_affine import \
    LowerCholeskyAffine


class OrnsteinUhlenbeck(Task):
    def __init__(self):
        """Ornstein-Uhlenbeck Process

        x_{t+1} = x_t + ∆x_t

        ∆xt = θ1(exp(θ2) − xt)∆t + 0.5ϵ, ϵ~𝒩(0,∆t)
        D = 50, ∆t = 0.2 and x0 = 10
        θ1 ∼ U(0, 1), θ1 ∼ U(−2.0, 2.0)
        """
        dim_data = 50

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000020,  # observation 1
            1000030,  # observation 2
            1000034,  # observation 3
            1000013,  # observation 4
            1000004,  # observation 5
            1000011,  # observation 6
            1000012,  # observation 7
            1000039,  # observation 8
            1000041,  # observation 9
            1000009,  # observation 10
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="Ornstein-Uhlenbeck",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,  # type: ignore
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )

        # Prior
        self.prior_params = {
            "low": torch.Tensor([0., -2.]),
            "high": torch.Tensor([1., 2.])
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return [r"$\theta_1$", r"$\theta_2$"]

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(
        self,
        max_calls: Optional[int] = None,
    ) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """
        def simulator(params: torch.Tensor):
            """
            Batched, posterior-sampling-friendly Ornstein-Uhlenbeck Process.


            Simulate an Ornstein-Uhlenbeck process with the following update equation
            x_{t+1} = x_t + ∆x_t

            Where:
                - ∆xt = θ1(exp(θ2) − xt)∆t + 0.5ϵ, ϵ~𝒩(0,∆t)
                - D = 50, ∆t = 0.1 and x0 = 12

            And return {x_t}_{t=1..50}

            A naive implementation of this process is straightforward. This implementation is
            slightly tricker since it accounts for:
                - being able to condition the model on observed variables x_t
                - simulating batches of observation in a vectorized manner.
            """
            num_timepoints = 50
            delta_t = torch.tensor(0.1)

            assert len(params.shape) == 2
            assert params.shape[1] == 2
            num_observations = params.shape[0]

            theta_1 = params[:, 0]
            theta_2 = params[:, 1]

            x0 = torch.tensor(12.0) * torch.ones(num_observations)
            y0 = x0 - theta_2.exp()
            z0 = y0

            power_mat = torch.arange(1, num_timepoints + 1).reshape(-1, 1) - torch.arange(
                1, num_timepoints + 1
            ).reshape(1, -1)

            power_mat = (1 - theta_1 * delta_t).reshape(-1, 1, 1).pow(power_mat.unsqueeze(0))
            power_mat = power_mat * 0.5 * delta_t.sqrt()

            power_mat = torch.tril(power_mat)

            bias = theta_2.exp().reshape(-1, 1) + z0.reshape(-1, 1) * (
                1 - delta_t * theta_1.reshape(-1, 1)
            ).pow(torch.arange(1, num_timepoints + 1).reshape(1, -1))

            orig_dist = pdist.Normal(
                torch.zeros(num_observations, num_timepoints),
                torch.ones(num_observations, num_timepoints),
            ).to_event(1)

            transformed_dist = TransformedDistribution(
                orig_dist, LowerCholeskyAffine(loc=bias, scale_tril=power_mat)
            )
            return pyro.sample("data", transformed_dist)


        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc

        Returns:
            Samples from reference posterior
        """
        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        if num_observation is not None:
            initial_params = self.get_true_parameters(num_observation=num_observation)
        else:
            initial_params = None

        proposal_samples = run_mcmc(
            task=self,
            kernel="Slice",
            jit_compile=False,
            num_warmup=10_000,
            num_chains=1,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            initial_params=initial_params,
            automatic_transforms_enabled=True,
        )

        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=True,
            density_estimator="flow",
            flow_model="nsf",
        )

        samples = run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=10_000,
            num_batches_without_new_max=1_000,
            multiplier_M=1.2,
            proposal_dist=proposal_dist, # type: ignore
        )

        return samples


if __name__ == "__main__":
    task = OrnsteinUhlenbeck()
    task._setup(n_jobs=-1)
