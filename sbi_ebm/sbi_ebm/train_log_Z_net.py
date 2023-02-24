# See issue #620.
# pytype: disable=wrong-keyword-args

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax import struct
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.training import train_state
from jax.nn import swish

from typing import Optional

from sbi_ebm.pytypes import Array, PRNGKeyArray, PyTreeNode

logging.set_verbosity(logging.INFO)


# XXX: should be called LZNetTrainingConfig
class LZNetConfig(struct.PyTreeNode):
    width: int = struct.field(pytree_node=False, default=100)
    depth: int = struct.field(pytree_node=False, default=4) # number of non-linearities basically
    max_iter: int = struct.field(pytree_node=False, default=500)
    learning_rate: float = struct.field(pytree_node=True, default=0.001)
    learn_grad: bool = struct.field(pytree_node=False, default=False)
    batch_size: int = struct.field(pytree_node=False, default=10000)
    select_based_on_test_loss: bool = struct.field(pytree_node=False, default=False)
    use_l1_loss: bool = struct.field(pytree_node=False, default=True)
    
    

class LogZMLP(Module):
    """A simple MLP model."""

    width: int = 100
    depth: int = 4

    @compact
    def __call__(self, x: Array) -> Array:
        if len(x.shape) > 2:
            raise ValueError("x must be a 1D or 2D array.")

        for _ in range(self.depth):
            x = Dense(features=self.width)(x)
            x = swish(x)
        x = Dense(1)(x)
        return x


# keep GradLogZMLP and GradLogZNet in case if needed
class GradLogZMLP(Module):
    """A simple MLP model."""

    width: int = 100
    depth: int = 3

    @compact
    def __call__(self, x: Array) -> Array:
        if len(x.shape) == 1:
            dim_x = x.shape[0]
        elif len(x.shape) == 2:
            dim_x = x.shape[1]
        else:
            raise ValueError("x must be a 1D or 2D array.")

        for _ in range(self.depth):
            x = Dense(features=self.width)(x)
            x = swish(x)
        x = Dense(dim_x)(x)
        return x


class LogZNet(struct.PyTreeNode):
    params: PyTreeNode
    config: LZNetConfig
    scale: float = struct.field(pytree_node=False, default=1.0)
    bias: Optional[Array] = struct.field(pytree_node=False, default=None)

    def predict(self, theta: Array) -> Array:
        if self.bias is None:
            return self.scale * LogZMLP(width=self.config.width, depth=self.config.depth).apply({"params": self.params}, theta)[..., 0]
        else:
            return self.scale * LogZMLP(width=self.config.width, depth=self.config.depth).apply({"params": self.params}, theta)[..., 0] + jnp.dot(self.bias, theta)


class GradLogZNet(struct.PyTreeNode):
    params: PyTreeNode
    config: LZNetConfig

    def predict(self, theta: Array) -> Array:
        return GradLogZMLP(width=self.config.width, depth=self.config.depth).apply({"params": self.params}, theta)


@jax.jit
def apply_model(
    state: train_state.TrainState, thetas: Array, grad_e_theta_x: Array, config: LZNetConfig
) -> Tuple[PyTreeNode, float]:
    """
    Given a conditional EBM p(x|theta; psi) = exp(-E_psi(x, theta)) / Z(theta, psi),
    estimate the the log partition function Z(theta, psi) (or its
    gradient) w.r.t. theta by minimizing the squared error between the gradient of
    the log partition function and the gradient of the energy function. I.e.
    match both sides of the relation
    grad_theta(log(Z(theta, psi))) = -E_x|theta(grad_theta(E_psi(x, theta))),
    which follows from the fact that Z(theta, psi) = int exp(-E_psi(x, theta)) dx.
    """
    if not config.learn_grad:
        def loss_fn(params):
            grad_log_z_pred = jax.vmap(jax.grad(lambda params,thetas: jnp.squeeze(LogZMLP(width=config.width, depth=config.depth).apply({"params": params}, thetas)), 1), (None, 0))(params, thetas)
            # E_p(x|theta) ∇_theta log Z(theta, \psi) = - E_p(x|theta) ∇_theta E_psi(x, theta)
            if config.use_l1_loss:
                print('using l1 loss')
                loss = jnp.average(jnp.sum(jnp.abs(grad_log_z_pred + grad_e_theta_x), axis=1))
            else:
                print('using l2 loss')
                loss = jnp.average(jnp.square(jnp.linalg.norm(grad_log_z_pred + grad_e_theta_x, axis=1)))
            return loss
    else:
        def loss_fn(params):
            grad_log_z_pred = GradLogZMLP(width=config.width, depth=config.depth).apply({"params": params}, thetas)
            # E_p(x|theta) ∇_theta log Z(theta, \psi) = - E_p(x|theta) ∇_theta E_psi(x, theta)
            if config.use_l1_loss:
                loss = jnp.average(jnp.sum(jnp.abs(grad_log_z_pred + grad_e_theta_x), axis=1))
            else:
                print('using l2 loss')
                loss = jnp.average(jnp.square(grad_log_z_pred + grad_e_theta_x))
            return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(
    state, train_ds, key, config: LZNetConfig, steps_per_epoch: int, len_training_data: int
) -> Tuple[train_state.TrainState, float]:
    """Train for a single epoch."""
   
    # TODO: doesn't take into account "skip incomplete batch" as in previous code--careful
    perms = jax.random.permutation(key, len_training_data)[:steps_per_epoch*config.batch_size]
    perms = perms.reshape((steps_per_epoch, config.batch_size))

    train_loss = 0.

    for perm in perms:
        batch_thetas = train_ds[0][perm, ...]
        batch_grad_e_theta_x = train_ds[1][perm, ...]
        grads, loss = apply_model(state, batch_thetas, batch_grad_e_theta_x, config)
        state = update_model(state, grads)
        train_loss += loss / len(perms)

    return state, train_loss


def create_train_state(key: PRNGKeyArray, X: Array, config: LZNetConfig, params: PyTreeNode):
    """Creates initial `TrainState`."""
    if not config.learn_grad:
        mlp = LogZMLP(config.width, config.depth)
    else:
        mlp = GradLogZMLP(config.width, config.depth)
    assert len(X.shape) == 2

    if params is None:
        key, subkey = jax.random.split(key)
        params = mlp.init(subkey, jnp.ones_like(X[0, :]))["params"]
    tx = optax.adamw(config.learning_rate, weight_decay=1e-5)
    return train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)


# Added `learn_grad`: if True, learn gradient of log(Z); otherwise, learn log(Z) as normal
def train_log_z_net(thetas: Array, grad_e_theta_x: Array, key: PRNGKeyArray, config: LZNetConfig, init_params: Optional[PyTreeNode]=None) -> LogZNet:

    from sklearn.model_selection import train_test_split

    (
        theta_train,
        theta_test,
        grad_e_theta_x_train,
        grad_e_theta_x_test,
    ) = train_test_split(thetas, grad_e_theta_x, random_state=43, train_size=0.8)
    assert not isinstance(theta_train, list)
    
    state = create_train_state(key, theta_train, config, init_params)

    config = config.replace(batch_size = min(config.batch_size, theta_train.shape[0]))
    steps_per_epoch = max(theta_train.shape[0] // config.batch_size, 1)
    
    jitted_train_epoch = jax.jit(train_epoch, static_argnums=(4,5,))
    prev_test_loss = 1e30
    prev_state = None

    for epoch in range(1, config.max_iter):
        key, subkey = jax.random.split(key)
        state, train_loss = jitted_train_epoch(
            state, (theta_train, grad_e_theta_x_train), subkey, config, steps_per_epoch, theta_train.shape[0]
        )
        _, test_loss = apply_model(state, theta_test, grad_e_theta_x_test, config)
        
        # TODO: this really should be more principled...see `likelihood_ebm.Trainer.train_ebm_likelihood_model`
        # i.e. number of warmup iterations, patience, etc. in config
        if config.select_based_on_test_loss:
            if epoch > 1:
                if test_loss > prev_test_loss:
                    print("test loss increased, stopping training at epoch ", str(epoch))
                    state = prev_state
                    break
            prev_test_loss = test_loss
            prev_state = state

        if (epoch % max(config.max_iter // 20, 1)) == 0:
            # logging.info(
            #     "epoch:% 3d, train_loss: %.6f, train_accuracy: %.4f, test_loss: %.6f, test_accuracy: %.4f"
            #     % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
            # )
            logging.info(
                "epoch:% 3d, train_loss: %.4f, test_loss: %.4f"
                % (epoch, train_loss, test_loss)
            )

    assert state is not None
    return LogZNet(state.params, config)


if __name__ == "__main__":
    pass
    # state, train_ds, test_ds = train_grad_log_z_net()
