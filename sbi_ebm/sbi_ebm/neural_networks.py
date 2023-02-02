import jax.numpy as jnp
from flax.linen.linear import Dense
from flax.linen.module import Module
from jax.nn import swish, tanh


class MLP(Module):
    width: int = 150
    depth: int = 4
    def setup(self):
        # self.layers = [Dense(self.width) for _ in range(4)] + [Dense(1, use_bias=False)]
        self.layers = [Dense(self.width) for _ in range(self.depth)] + [Dense(1, use_bias=False)]
        # self.layers = [Dense(self.width) for _ in range(2)] + [Dense(1, use_bias=False)]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = swish(x)
        return x


class IceBeem(Module):
    def setup(self):
        self.x_net = [Dense(50) for _ in range(3)]
        self.z_net = [Dense(50) for _ in range(3)]

    def __call__(self, inputs):
        z, x = inputs
        for i, lyr in enumerate(self.x_net):
            x = lyr(x)
            if i != len(self.x_net) - 1:
                x = tanh(x)

        for i, lyr in enumerate(self.z_net):
            z = lyr(z)
            if i != len(self.z_net) - 1:
                z = tanh(z)
        return jnp.dot(z, x)
