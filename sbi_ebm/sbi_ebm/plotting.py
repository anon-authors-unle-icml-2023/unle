from types import FunctionType
from typing import Optional, Tuple, Union

import deneb as den
import jax
import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.numpy import vectorize
from matplotlib.pyplot import Figure
from numpyro import distributions as np_distributions
from sbibm.visualisation import fig_posterior as sbibm_fig_posterior

from sbi_ebm.distributions import (DoublyIntractableJointLogDensity,
                                   ThetaConditionalLogDensity)
from sbi_ebm.pytypes import (Array, LogDensity_T, LogLikelihood_T, Numeric,
                             PRNGKeyArray)
from sbi_ebm.sbi_ebm import EBMPosterior
from sbi_ebm.sbibm.sbi_ebm import TrainEvalTresults
from sbi_ebm.sbibm.tasks import JaxTask, get_task


def fig_posterior_no_sampling(
    posterior: Union[np_distributions.Distribution, LogDensity_T],
    num_bins: Optional[int] = 20,
    z_scored_space=False,
    log_space=False,
    ylim=None,
    bounds=None,
    ret: Optional[TrainEvalTresults] = None,
    task_name: Optional[str] = None,
    num_observation: Optional[int] = None,
    _f: float = 0.,
    posterior_samples: Optional[Array] = None,
) -> Figure:
    if bounds is None:
        if task_name is not None:
            task = get_task(task_name)
        elif ret is not None:
            task_name = ret.train_results.config.task.task_name
            task = get_task(ret.train_results.config.task.task_name)
            num_observation = ret.train_results.config.task.num_observation
        else:
            raise ValueError("Either task_name, ret or bounds must be provided to draw a posterior figure")


        if posterior_samples is None:
            assert task_name is not None
            assert num_observation is not None

            _ref_post_samples =  task.get_reference_posterior_samples(num_observation)
            ref_post_samples = jnp.array(_ref_post_samples)
        else:
            ref_post_samples = posterior_samples

        # if z_scored_space:
        #     assert ret is not None
        #     ref_post_samples = z_transform.inv(ref_post_samples)


        bounds =  tuple(zip(jnp.min(ref_post_samples,axis=0) - _f*jnp.abs(jnp.min(ref_post_samples, axis=0))  , jnp.max(ref_post_samples, axis=0) + _f *jnp.abs(jnp.max(ref_post_samples,axis=0))))
        print(bounds)

        # print("ref mean", ref_post_samples.mean(axis=0))
        # print("unle mean", jnp.mean(ret.train_results.single_round_results[round].posterior_samples, axis=0))
    if num_bins is None:
        if task_name == "LDCT":
            num_bins = 100
        elif task_name == "slcp":
            # a = 25 * 25 * 25 * 25
            # bounds=((-3, 3),(-3, 3),(-3, 3),(-3, 3),(-3, 3))
            num_bins = 15
        elif task_name == "lotka_volterra":
            # bounds=((0, 4),(0, 0.4),(-0, 3),(0, 0.3))
            # bounds=((0.01, 3),(0.1, 0.3),(0.2, 2),(0.1, 0.3))
            # bounds =  _get_smart_bounds()
            num_bins = 25
        else:
            raise ValueError

    assert isinstance(bounds, tuple)
    assert bounds is not None
    dim_params = len(bounds)

    # integrate_domains = jnp.linspace(-2.9, 2.9, 20)
    _1d_grids = [jnp.linspace(min_, max_, num_bins) for min_, max_ in bounds]
    grids = jnp.meshgrid(*_1d_grids, indexing="ij")
    grid = jnp.stack(grids, axis=-1)

    assert posterior is not None
    if isinstance(posterior, np_distributions.Distribution):
        if z_scored_space:
            assert isinstance(posterior, EBMPosterior)
            lp = posterior.log_prob_zscored_space
        else:
            lp = posterior.log_prob
    elif hasattr(posterior, "__call__"):
        lp = posterior
    else:
        raise ValueError

    # broadcast computation of logprob over all but the last dimension
    vals = vectorize(lp, signature="(k)->()")(grid)

    f, axs = plt.subplots(nrows=dim_params, ncols=dim_params, figsize=(10, 10))
    for i in range(dim_params):
        for j in range(dim_params):

            axes_to_sum = [k for k in range(dim_params) if k not in (i, j)]

            if len(axes_to_sum) > 0:
                estimated_2d_energy = jax.nn.logsumexp(vals, axis=axes_to_sum)
            else:
                estimated_2d_energy = vals

            estimated_2d_energy = estimated_2d_energy - jnp.max(estimated_2d_energy)

            if ylim is not None:
                estimated_2d_energy = jnp.minimum(estimated_2d_energy, ylim[1])
                estimated_2d_energy = jnp.maximum(estimated_2d_energy, ylim[0])

            if not log_space:
                estimated_2d_energy = jnp.exp(estimated_2d_energy)

            ax = axs[j, i]
            if i == j:
                ax.plot(_1d_grids[i], estimated_2d_energy)
                if ylim is not None:
                    ax.set_ylim(ylim)
                continue

            if i > j:
                estimated_2d_energy = estimated_2d_energy.T

            grid2d = jnp.meshgrid(_1d_grids[i], _1d_grids[j], indexing="ij")

            ax.pcolor(grid2d[0], grid2d[1], estimated_2d_energy, shading="auto")
    return f


def _make_histogram(
    log_prob: LogDensity_T, bounds: tuple[tuple[int, int], ...], num_bins: int
) -> Tuple[Array, Array]:
    # integrate_domains = jnp.linspace(-2.9, 2.9, 20)
    _1d_grids = [jnp.linspace(min_, max_, num_bins) for min_, max_ in bounds]
    grids = jnp.meshgrid(*_1d_grids, indexing="ij")
    grid = jnp.stack(grids, axis=-1)

    # broadcast computation of logprob over all but the last dimension
    vals = vectorize(log_prob, signature="(k)->()")(grid)

    # subtract crude estimate of the log-normalization constant
    vals = vals - jax.nn.logsumexp(vals)
    return jnp.exp(vals), grid


def density_difference(
    log_prob_1: LogDensity_T,
    log_prob_2: LogDensity_T,
    bounds: tuple[tuple[int, int], ...],
    num_bins: int,
    plot: bool = False,
    title_extra: str = "",
) -> Numeric:
    dim_params = len(bounds)
    vals_1, _ = _make_histogram(log_prob_1, bounds, num_bins)
    vals_2, _ = _make_histogram(log_prob_2, bounds, num_bins)

    if plot:
        f, axs = plt.subplots(nrows=dim_params, ncols=dim_params, figsize=(10, 10))
    else:
        f, axs = None, None

    for i in range(dim_params):
        for j in range(dim_params):
            _1d_grid_i = jnp.linspace(bounds[i][0], bounds[i][1], num_bins)
            _1d_grid_j = jnp.linspace(bounds[j][0], bounds[j][1], num_bins)

            dx_i = _1d_grid_i[1] - _1d_grid_i[0]
            dx_j = _1d_grid_j[1] - _1d_grid_j[0]
            dxi_dxj = dx_i * dx_j

            axes_to_sum = tuple([k for k in range(dim_params) if k not in (i, j)])

            marginal_density_1 = jnp.sum(vals_1, axis=axes_to_sum) / dxi_dxj
            marginal_density_2 = jnp.sum(vals_2, axis=axes_to_sum) / dxi_dxj

            marginal_density_diff = marginal_density_1 - marginal_density_2

            if plot:
                assert axs is not None  # type narrowing
                assert f is not None  # type narrowing
                ax = axs[j, i]  # type: ignore
                if i == j:
                    x_input = jnp.linspace(bounds[i][0], bounds[i][1], num_bins)

                    ax.plot(x_input, marginal_density_1, label="1")
                    ax.plot(x_input, marginal_density_2, label="2")
                    ax.plot(
                        x_input, marginal_density_1 - marginal_density_2, label="diff"
                    )

                    # ax.set_ylim(-1, 1)
                    ax.legend()

                else:
                    if i > j:
                        marginal_density_diff = marginal_density_diff.T
                    grid2d = jnp.meshgrid(_1d_grid_i, _1d_grid_j, indexing="ij")
                    cax = ax.pcolor(
                        grid2d[0], grid2d[1], marginal_density_diff, shading="auto"
                    )
                    _ = f.colorbar(mappable=cax, ax=ax)
                    # cbar.set_label("Difference")

                ax.set_title(
                    f"marginal difference {dxi_dxj * marginal_density_diff.sum():.4f}"
                )

    total_diff = jnp.sum(jnp.abs(vals_1 - vals_2))
    if plot:
        assert f is not None
        f.suptitle(f"Total difference: {total_diff:.4f} {title_extra}")

    return total_diff


def average_density_difference(
    log_lik_1: LogLikelihood_T,
    log_lik_2: LogLikelihood_T,
    prior: np_distributions.Distribution,
    bounds: tuple[tuple[int, int], ...],
    num_bins: int,
    num_theta_draws: int,
    key: PRNGKeyArray,
):
    theta_draws = prior.sample(key, sample_shape=(num_theta_draws,))

    vmapped_log_likelihoods_1 = ThetaConditionalLogDensity(log_lik_1, theta_draws)
    vmapped_log_likelihoods_2 = ThetaConditionalLogDensity(log_lik_2, theta_draws)

    from jax import vmap

    vmapped_density_diff_func = vmap(
        density_difference,
        in_axes=(ThetaConditionalLogDensity(None, 0),
                 ThetaConditionalLogDensity(None, 0), None, None),  # type: ignore
    )

    vmapped_density_diff_vals = vmapped_density_diff_func(
        vmapped_log_likelihoods_1, vmapped_log_likelihoods_2, bounds, num_bins
    )
    return jnp.average(vmapped_density_diff_vals)


def fig_posterior(ret: TrainEvalTresults, round: int = -1, **kwargs):
    t = get_task(ret.train_results.config.task.task_name)

    f = sbibm_fig_posterior(
        task=t,
        num_observation=ret.train_results.config.task.num_observation,
        samples_tensor=torch.from_numpy(
            np.array(ret.train_results.single_round_results[round].posterior_samples)
        )[:1000],
        num_samples=min(
            1000, len(ret.train_results.single_round_results[round].posterior_samples)
        ),
        task_name=ret.train_results.config.task.task_name,
        **kwargs,
    )
    return f


# def fig_posterior_exact(ret: TrainEvalTresults, round: int = -1, **kwargs):
#     from sbi_ebm.sbibm.tasks import get_task
#     _f = 0.1
#     _ref_post_samples =  get_task(ret.train_results.config.task.task_name).get_reference_posterior_samples(ret.train_results.config.task.num_observation)
#     ref_post_samples = jnp.array(_ref_post_samples)
# 
#     def _get_smart_bounds():
#         bounds =  tuple(zip(jnp.min(ref_post_samples,axis=0) - _f*jnp.abs(jnp.min(ref_post_samples, axis=0))  , jnp.max(ref_post_samples, axis=0) + _f *jnp.abs(jnp.max(ref_post_samples,axis=0))))
#         return bounds
# 
#     print("ref mean", ref_post_samples.mean(axis=0))
#     print("unle mean", jnp.mean(ret.train_results.single_round_results[round].posterior_samples, axis=0))
# 
#     if ret.train_results.config.task.task_name == "LDCT":
#         num_bins = 100
#     elif ret.train_results.config.task.task_name == "slcp":
#         a = 25 * 25 * 25 * 25
#         bounds=((-3, 3),(-3, 3),(-3, 3),(-3, 3),(-3, 3))
#         num_bins = 15
#     elif ret.train_results.config.task.task_name == "lotka_volterra":
#         # bounds=((0, 4),(0, 0.4),(-0, 3),(0, 0.3))
#         bounds=((0.01, 3),(0.1, 0.3),(0.2, 2),(0.1, 0.3))
#         bounds =  _get_smart_bounds()
#         num_bins = 25
#     else:
#         raise ValueError
# 
# 
#     f = fig_posterior_no_sampling(
#         ret,
#         ret.train_results.config.task.task_name,
#         num_bins,
#         # ret.train_results.single_round_results[-1].get_posterior(99),
#         ret.train_results.single_round_results[-1].get_posterior(99),
#         z_scored_space=False,
#         log_space=False,
#         # bounds=((-9.99, -6.01), (-4, 1.8)),
#         # bounds=((-9.99, -6.01), (-4, 1.8)),
#         # bounds=((-3, 3), (-3, 3)),
#         bounds=bounds,
#         # bounds=((-3, 3),(-3, 3),(-3, 3),(-3, 3),(-3, 3)),
#         # ylim=(-10, 0)
#     )
#     f.suptitle("doubly unnormalized posterior values")
