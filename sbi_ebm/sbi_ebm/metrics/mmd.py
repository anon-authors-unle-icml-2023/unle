from typing import Optional
from jax.tree_util import tree_map

import jax.numpy as jnp
from flax import struct

from sbi_ebm.pytypes import Array, Numeric
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation


class GaussianKernel(struct.PyTreeNode):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Numeric:
        return jnp.exp(-jnp.dot(x - y, x - y) / (2 * self.sigma ** 2))

    def mmd(
        self,
        X: Array,
        Y: Array,
        ws_x: Optional[Array] = None,
        ws_y: Optional[Array] = None,
    ) -> Numeric:
        assert len(X.shape) == len(Y.shape) == 2
        assert X.shape[1] == Y.shape[1]
        nx, ny = X.shape[0], Y.shape[0]

        if ws_x is None:
            _ws_x = jnp.ones((nx,)) / nx
        else:
            _ws_x = ws_x

        if ws_y is None:
            _ws_y = jnp.ones((ny,)) / ny
        else:
            _ws_y = ws_y

        assert len(_ws_x.shape) == 1
        assert _ws_x.shape[0] == X.shape[0]

        assert len(_ws_y.shape) == 1
        assert _ws_y.shape[0] == Y.shape[0]

        _km_func = jnp.vectorize(self.__call__, signature="(k),(k)->()")

        _kxx = (_ws_x @ _km_func(X[:, None, :], X[None, :, :])) @ _ws_x
        _kyy = (_ws_y @ _km_func(Y[:, None, :], Y[None, :, :])) @ _ws_y
        _kxy = (_ws_x @ _km_func(X[:, None, :], Y[None, :, :])) @ _ws_y

        return _kxx + _kyy - 2 * _kxy

    def mmd_pa(self, X: ParticleApproximation, Y: ParticleApproximation) -> Numeric:
        return self.mmd(X=X.xs, Y=Y.xs, ws_x=X.ws, ws_y=Y.ws)


def mmd(
    X: Array, Y: Array, _ws_x: Optional[Array] = None, _ws_y: Optional[Array] = None
) -> Numeric:
    return GaussianKernel(sigma=1).mmd(X, Y, _ws_x, _ws_y)


def mmd_pa(X: ParticleApproximation, Y: ParticleApproximation) -> Numeric:
    X = tree_map(lambda x: x[:10000], X).ensure_normalized_weights()
    Y = tree_map(lambda x: x[:10000], Y).ensure_normalized_weights()
    return GaussianKernel(sigma=1).mmd_pa(X=X, Y=Y)
