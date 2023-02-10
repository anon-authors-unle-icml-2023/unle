from typing import Any, Callable, Union
from typing_extensions import TypeAlias
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

try:
    import torch
except ImportError:
    torch = None
    print("PyTorch not installed, pytorch plotting functions will not work")


import jax
from jax import jit, vmap
from flax import struct



Array_T = Any
Scalar_T: TypeAlias = Union[Array_T, float, int]
Density_T: TypeAlias = Callable[[Array_T], Scalar_T]

class ConditionnedDensity(struct.PyTreeNode):
    theta: Any
    log_prob: Any = struct.field(pytree_node=False)

    def __call__(self, x):
        return self.log_prob(self.theta, x)



def make_unnormalized_kde_density(points, bandwidth, weights=None):
    if weights is None:
        weights = jnp.ones(points.shape[0]) / points.shape[0]

    def gaussian_kernel(x, point, weight):
        return weight * jnp.exp(-0.5 * jnp.sum(jnp.square((x - point) / bandwidth)))

    vmapped_gaussian_kernel = vmap(
        gaussian_kernel, in_axes=(None, 0, 0)  # type: ignore
    )

    def unnormalized_kde_density(x):
        return jnp.sum(vmapped_gaussian_kernel(x, points, weights))

    return unnormalized_kde_density


# XXX: only works for 2-dimensional input space
def plot_densities(densities_dict, bounds, nbins=100, axs=None, log_space=False, return_log_space=False, batch_size=-1):
    (x_min, x_max), (y_min, y_max) = bounds
    num_densities = len(densities_dict)

    if axs is None:
        _, axs = plt.subplots(ncols=num_densities, figsize=(num_densities * 5, 5))
        if num_densities == 1:
            axs = [axs]

    _X, _Y = jnp.meshgrid(
        jnp.linspace(x_min, x_max, nbins), jnp.linspace(y_min, y_max, nbins),
        indexing="ij"
    )
    _inputs = jnp.stack((_X, _Y), axis=-1)

    for ax, (_, density) in zip(axs, densities_dict.items()):
        grid_mapped_density = jit(vmap(vmap(density, in_axes=0), in_axes=0))
        if log_space:
            if batch_size > 0: 
                one_batch = jnp.array_split(_inputs, len(_inputs) // batch_size)[0]
                print(len(one_batch))
                val = grid_mapped_density(one_batch)
                vals = jnp.concatenate([grid_mapped_density(batch) for batch in jnp.array_split(_inputs, len(_inputs) // batch_size)])
            else:
                vals = grid_mapped_density(_inputs)
            vals = jnp.exp(vals - jnp.max(vals))
        else:
            assert batch_size == -1
            vals = grid_mapped_density(_inputs)
        vals /= vals.sum()

        if return_log_space:
            vals = jnp.log(vals)
        ax.pcolor(_X, _Y, vals, shading="auto")


def plot_density_pytorch(bounds, density, nbins=100, f=None, ax=None, log_space=False):
    assert torch is not None
    ((xmin, xmax), (ymin, ymax)) = bounds

    if ax is None:
        f, ax = plt.subplots()

    xi, yi = np.mgrid[xmin : xmax : nbins * 1j, ymin : ymax : nbins * 1j]

    xs = torch.Tensor(np.vstack([xi.flatten(), yi.flatten()])).T
    xs.requires_grad = False

    vals = []
    for x in xs:
        vals.append(density(x))

    vals = torch.Tensor(vals)

    if log_space:
        print('reomving max likelihood val for numerical stability')
        vals = (vals - vals.max()).exp()

    if ax is None:
        f, ax = plt.subplots()

    # Make the plot
    cax = ax.pcolormesh(xi, yi, vals.reshape(xi.shape))
    if f is not None:
        f.colorbar(cax, ax=ax)
    return vals


def _make_histogram(d: Density_T, bounds: Tuple[Tuple[int, int], ...], nbins: int, log_space=False, return_log_space=False) -> Array_T:
    (x_min, x_max), (y_min, y_max) = bounds
    _X, _Y = jnp.meshgrid(
        jnp.linspace(x_min, x_max, nbins), jnp.linspace(y_min, y_max, nbins),
        indexing="ij"
    )
    _inputs = jnp.stack((_X, _Y), axis=-1)

    dx, dy = _X[0, 1] - _X[0, 0], _Y[1, 0] - _Y[0, 0]
    dxdy = dx * dy

    grid_mapped_d = jit(vmap(vmap(d, in_axes=0), in_axes=0))

    if not return_log_space:
        vals = grid_mapped_d(_inputs)
        if log_space:
            vals = jnp.exp(vals - jnp.max(vals))
        vals /= vals.sum()
    else:
        vals = grid_mapped_d(_inputs)
        if log_space:
            from jax.nn import logsumexp
            vals = vals - logsumexp(vals)
        else:
            vals /= vals.sum()
            vals = jnp.log(vals)
    return vals


def _make_normalized_density(d: Density_T, bounds: Tuple[Tuple[int, int], ...], nbins: int, log_space=False, return_log_space=False) -> Array_T:
    (x_min, x_max), (y_min, y_max) = bounds
    _X, _Y = jnp.meshgrid(
        jnp.linspace(x_min, x_max, nbins), jnp.linspace(y_min, y_max, nbins),
        indexing="ij",
    )
    _inputs = jnp.stack((_X, _Y), axis=-1)

    dx, dy = jnp.abs(_X[0, 0] - _X[1, 0]), jnp.abs(_Y[0, 0] - _Y[0, 1])
    dxdy = dx * dy


    grid_mapped_d = jit(vmap(vmap(d, in_axes=0), in_axes=0))

    if not return_log_space:
        vals = grid_mapped_d(_inputs)
        if log_space:
            vals = jnp.exp(vals - jnp.max(vals))
        vals /= (vals.sum() + 1e-10)
        vals /= dxdy

    else:
        vals = grid_mapped_d(_inputs)
        if log_space:
            from jax.nn import logsumexp
            vals = vals - logsumexp(vals)
        else:
            vals /= (vals.sum() + 1e-10)
            vals = jnp.log(vals)
        vals -= jnp.log(dxdy)
    return vals


def density_l2_dist(d1, d2, bounds, nbins=100):
    vals_1 = _make_histogram(d1, bounds, nbins)
    vals_2 = _make_histogram(d2, bounds, nbins)
    return jnp.sqrt(jnp.sum(jnp.square(vals_1 - vals_2)))


def normalize_density(density, bounds, nbins=100, log_space=True):
    (x_min, x_max), (y_min, y_max) = bounds
    _X, _Y = jnp.meshgrid(
        jnp.linspace(x_min, x_max, nbins), jnp.linspace(y_min, y_max, nbins),
        indexing="ij",
    )
    _inputs = jnp.stack((_X, _Y), axis=-1)
    def normalized_density(x):
        from jax.nn import logsumexp
        if log_space:
            conditioned_log_density_vals = vmap(vmap(density))(_inputs)
            log_Z = logsumexp(conditioned_log_density_vals)
            return density(x) - log_Z
        else:
            conditioned_density_vals = vmap(vmap(density))(_inputs)
            return density(x) / jnp.sum(conditioned_density_vals)
    return normalized_density


def normalize_posterior(prior, likelihood, x_obs, bounds, nbins=100, log_space=True):
    def normalized_posterior(theta):
        conditioned_likelihood = ConditionnedDensity(theta, likelihood)
        # normalized_likelihood = normalize_density(conditioned_likelihood, bounds, nbins, log_space)
        normalized_likelihood = normalize_density(conditioned_likelihood, ((theta[0] - 3, theta[0] +3), (theta[1] - 3, theta[1] +3)), nbins, log_space)
        return prior(theta) + normalized_likelihood(x_obs)
    return normalized_posterior



def plot_doubly_intractable_posterior(prior, likelihood, bounds, nbins=100, axs=None, log_space=False, return_log_space=False):
    (theta1_min, theta1_max), (theta2_min, theta2_max), (x1_min, x1_max), (x2_min, x2_max) = bounds
    # normalize the likelihood for each theta.
    _thetas1, _thetas2 = jnp.meshgrid(
        jnp.linspace(theta1_min, theta1_max, nbins), jnp.linspace(theta2_min, theta2_max, nbins), indexing="ij"
    )
    _theta_inputs = jnp.stack((_thetas1, _thetas2), axis=-1)



    conditionned_densities = vmap(vmap(ConditionnedDensity, in_axes=(0, None)), in_axes=(0, None))(_theta_inputs, likelihood)

    # p(x|theta) for all x, theta. Normalized for each theta.
    histograms_likelihood = vmap(vmap(_make_normalized_density, in_axes=(ConditionnedDensity(0, likelihood), None, None, None, None)), in_axes=(ConditionnedDensity(0, likelihood), None, None, None, None))(
        conditionned_densities, bounds[2:], nbins, log_space, return_log_space
    )
    histograms_prior = _make_normalized_density(prior, bounds[:2], nbins, log_space=log_space, return_log_space=return_log_space)


    if return_log_space:
        from jax.nn import logsumexp
        posterior_vals = histograms_prior[:, :, None, None] + histograms_likelihood
        # posterior_vals = joint -  logsumexp(joint, axis=(0, 1), keepdims=True)
    else:
        posterior_vals = histograms_prior[:, :, None, None] * histograms_likelihood
        # posterior_vals = joint / (jnp.sum(joint, axis=(0, 1), keepdims=True) + 1e-10)

    return conditionned_densities, posterior_vals
