from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import random
from jax._src.api import vmap
from jax.nn import log_softmax, logsumexp
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree  # type: ignore
from numpyro import distributions as np_distributions
import numpy as np

from sbi_ebm.pytypes import Array, Numeric, PRNGKeyArray


class MOGDistribution(np_distributions.Distribution):
    def __init__(self, cluster_means: Array, cluster_covs: Array, cluster_props: Array):
        assert len(cluster_means.shape) == 2
        self._num_clusters, self._num_dims = cluster_means.shape
        self.cluster_means = cluster_means

        assert len(cluster_props.shape) == 1
        self.cluster_props = cluster_props

        self.cluster_covs = cluster_covs

        self._dists = np_distributions.MultivariateNormal(
            cluster_means, covariance_matrix=cluster_covs
        )

        super(MOGDistribution, self).__init__(
            batch_shape=(), event_shape=(self._num_dims,)
        )

    def log_prob(self, x: Array):
        assert len(x.shape) == 1
        return logsumexp(self._dists.log_prob(x), b=self.cluster_props)

    def _sample_from_cluster_idx(self, key, idx):
        return tree_map(lambda d: d[idx], self._dists).sample(key)

    def sample(self, key: PRNGKeyArray, sample_shape: tuple = ()) -> Array:
        if sample_shape == tuple():
            sample_shape = (1,)

        key, key_latent = random.split(key)

        mn = np_distributions.Categorical(probs=self.cluster_props)
        idxs = mn.sample(key_latent, sample_shape=sample_shape)
        keys_observed = random.split(key, num=sample_shape[0])
        return vmap(self._sample_from_cluster_idx, in_axes=(0, 0))(  # type: ignore
            keys_observed, idxs
        )


class MOGResult(NamedTuple):
    # don't return a MOGDistribution because numpyro Distributions objects are not
    # vmap-able
    min_std: float
    cluster_init: Array
    cluster_means: Array
    cluster_covs: Array
    cluster_props: Array
    log_probs: Array
    final_log_prob: Numeric
    converged: bool
    num_iter_convergence: Numeric

    def to_dist(self) -> MOGDistribution:
        return MOGDistribution(
            self.cluster_means, self.cluster_covs, self.cluster_props
        )


def _kmeans_plus_plus_init(data: Array, num_clusters: int, key: PRNGKeyArray) -> Array:
    num_points, num_dim = data.shape
    clusters = jnp.empty((num_clusters, num_dim))

    init_cluster_data_idx = random.choice(key, a=num_points)
    init_cluster_center = data[init_cluster_data_idx, :]

    this_cluster_center = init_cluster_center
    clusters = clusters.at[0, :].set(this_cluster_center)

    all_sq_dists = jnp.inf * jnp.ones((num_points, num_clusters))

    for i in range(num_clusters - 1):
        sq_dists = jnp.sum(jnp.square(data - this_cluster_center), axis=1)

        all_sq_dists = all_sq_dists.at[:, i].set(sq_dists)
        min_sq_dists = jnp.min(all_sq_dists, axis=1)

        key, subkey = random.split(key)
        next_cluster_idx = random.categorical(
            subkey, logits=jnp.log(min_sq_dists + 1e-15)
        )
        next_cluster_center = data[next_cluster_idx]

        this_cluster_center = next_cluster_center
        clusters = clusters.at[i + 1, :].set(this_cluster_center)

    return clusters


def _fit_one_mog(
    data: Array, num_clusters: int, min_std: float, max_iter: int,
    max_train_samples: int, cov_reg_param: float, key: PRNGKeyArray
) -> MOGResult:
    num_points, num_dims = data.shape

    assert len(data.shape) == 2
    # TODO: kmeans++?

    if data.shape[0] > max_train_samples:
        key, subkey = random.split(key)
        data = random.permutation(subkey, data, axis=0)
        data = data[:max_train_samples]

    # init_cluster_data_idx = random.choice(key, a=num_points, shape=(num_clusters,))

    # cluster_means = data[init_cluster_data_idx]
    key, key_init = random.split(key)
    init_cluster_means = _kmeans_plus_plus_init(data, num_clusters, key_init)
    init_cluster_covs = jnp.stack(
        [jnp.eye(num_dims) for _ in range(num_clusters)], axis=0
    )

    log_cluster_props = -np.log(num_clusters) * jnp.ones((num_clusters,))

    log_prob = prev_log_prob = -jnp.inf

    iter_no = 0
    assert max_iter > 0

    converged = False
    num_iter_convergence = 0

    log_probs = jnp.empty((max_iter,))

    cluster_means = init_cluster_means
    cluster_covs = init_cluster_covs


    for iter_no in range(max_iter):
        dists = np_distributions.MultivariateNormal(
            cluster_means,
            covariance_matrix=cluster_covs,
        )
        log_joint = dists.log_prob(data[:, None, :]) + log_cluster_props[None, :]

        log_prob = logsumexp(log_joint, axis=1).mean()
        log_probs = log_probs.at[iter_no].set(log_prob)

        # assert log_prob - prev_log_prob > -1e-6
        converged = jnp.abs(log_prob - prev_log_prob) < 1e-4
        num_iter_convergence += 1 - converged

        # E-step: compute posterior p(z=k|x) (num_points, num_clusters)
        log_resps = log_softmax(log_joint, axis=1)

        # M-step
        # data: (num_points, dim)
        # normalized_data_weights: (num_points,num_clusters)
        normalized_data_weights = jax.nn.softmax(log_resps, axis=0)
        cluster_means = jnp.sum(data[:, None, :] * normalized_data_weights[:, :, None], axis=0)

        log_cluster_props = jax.nn.log_softmax(jax.nn.logsumexp(log_resps, axis=0))

        def _compute_cov(mean, weights):
            if len(jnp.asarray(cov_reg_param).shape) == 1:
                return (
                    (data - mean).T @ jnp.diag(weights) @ (data - mean) + jnp.diag(cov_reg_param)
                )
            elif len(jnp.asarray(cov_reg_param).shape) == 0:
                return (
                    (data - mean).T @ jnp.diag(weights) @ (data - mean) + cov_reg_param * jnp.eye(num_dims)
                )
            else:
                raise ValueError

        cluster_covs = vmap(_compute_cov, in_axes=(0, 1))(cluster_means, normalized_data_weights)  # type: ignore

        prev_log_prob = log_prob

    def smooth_cov(cov_mat):
        from jax.numpy.linalg import eigh
        eigvals, eigvecs = eigh(cov_mat)
        return eigvecs @ jnp.diag(jnp.clip(jnp.real(eigvals), a_min=min_std ** 2)) @ eigvecs.T

    cluster_covs = vmap(smooth_cov)(cluster_covs)
    return MOGResult(
        min_std,
        init_cluster_means,
        cluster_means,
        cluster_covs,
        jax.nn.softmax(log_cluster_props),
        log_probs,
        log_prob,
        converged,
        num_iter_convergence,
    )


class MOGTrainingConfig(struct.PyTreeNode):
    num_clusters: int = struct.field(pytree_node=False)
    max_iter: int = struct.field(pytree_node=False)
    num_inits: int = struct.field(pytree_node=False)
    min_std: float = struct.field(pytree_node=True, default=0.01)
    max_train_samples: int = struct.field(pytree_node=False, default=1000)
    cov_reg_param: int = struct.field(pytree_node=False, default=1e-6)


def fit_mog(data: Array, config: MOGTrainingConfig, key: PRNGKeyArray, auto_select_num_clusters: bool = False) -> MOGResult:
    if not auto_select_num_clusters:
        keys = random.split(key, num=config.num_inits)
        vmapped_fit = vmap(_fit_one_mog, in_axes=(None, None, None, None, None, None, 0))  # type: ignore
        rets = vmapped_fit(data, config.num_clusters, config.min_std, config.max_iter,
                           config.max_train_samples, config.cov_reg_param, keys)

        print(rets.final_log_prob)
        best_fit_idx = jnp.argmax(rets.final_log_prob)
        return tree_map(lambda l: l[best_fit_idx], rets)
    else:
        return _fit_bayesian_mog(data, config, key)



def _fit_bayesian_mog(data: Array, config: MOGTrainingConfig, key: PRNGKeyArray):
    from sklearn.mixture import BayesianGaussianMixture
    bgm = BayesianGaussianMixture(
        n_components=config.num_clusters, n_init=config.num_inits, max_iter=config.max_iter,
        tol=1e-6, reg_covar=0.001)
    bgm.fit(np.array(data))

    bgm.init_params

    # select components accounting for 99.9% of the data
    num_selected_components = (1 - (1 - jnp.sort(bgm.weights_)[::-1].cumsum() < 1e-3)).sum()  # type: ignore
    idxs = jnp.argsort(bgm.weights_)[::-1][:num_selected_components].tolist()

    cluster_means = jnp.array(bgm.means_[idxs])
    cluster_covs = jnp.array(bgm.covariances_[idxs])

    cluster_weights = jnp.array(bgm.weights_[idxs])
    cluster_weights = cluster_weights / jnp.sum(cluster_weights)

    def smooth_cov(cov_mat):
        from jax.numpy.linalg import eigh
        eigvals, eigvecs = eigh(cov_mat)
        return eigvecs @ jnp.diag(jnp.clip(jnp.real(eigvals), a_min=config.min_std ** 2)) @ eigvecs.T

    cluster_covs = vmap(smooth_cov)(cluster_covs)

    return MOGResult(
        config.min_std,
        cluster_means,
        cluster_means,
        cluster_covs,
        cluster_weights,
        None,
        None,
        bgm.converged_,
        bgm.n_iter_,
    )
