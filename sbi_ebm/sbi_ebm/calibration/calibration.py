# See issue #620.
# pytype: disable=wrong-keyword-args

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.training import train_state
from jax.nn import relu

from sbi_ebm.pytypes import Array

logging.set_verbosity(logging.INFO)


class CalibrationMLP(Module):
    """A simple MLP model."""

    num_neurons: int = 200

    @compact
    def __call__(self, x):
        x = Dense(features=self.num_neurons)(x)
        x = relu(x)
        x = Dense(features=self.num_neurons)(x)
        x = relu(x)
        x = Dense(features=self.num_neurons)(x)
        x = relu(x)
        x = Dense(2)(x)
        return x


@jax.jit
def apply_model(state, images, labels, class_weights):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = CalibrationMLP().apply({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 2)
        class_weights_arr = class_weights[1] * (labels == 1) + class_weights[0] * (
            labels == 0
        )

        # loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))  # type: ignore
        loss = jnp.average(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot),  # type: ignore
            weights=class_weights_arr,
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    accuracy_0 = jax.lax.cond(
        (labels == 0).sum() > 0,
        lambda: ((jnp.argmax(logits, -1) == labels) * (labels == 0)).sum()
        / (labels == 0).sum(),
        lambda: 1.0,
    )

    accuracy_1 = jax.lax.cond(
        (labels == 1).sum() > 0,
        lambda: ((jnp.argmax(logits, -1) == labels) * (labels == 1)).sum()
        / (labels == 1).sum(),
        lambda: 1.0,
    )
    # print(accuracy)
    return grads, loss, (accuracy, accuracy_0, accuracy_1)


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng, class_weights):
    """Train for a single epoch."""
    train_ds_size = len(train_ds[0])
    steps_per_epoch = max(train_ds_size // batch_size, 1)
    # print(steps_per_epoch)

    perms = jax.random.permutation(rng, len(train_ds[0]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    # print(perms)
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    epoch_a0 = []
    epoch_a1 = []

    for perm in perms:
        batch_images = train_ds[0][perm, ...]
        batch_labels = train_ds[1][perm, ...]
        grads, loss, (accuracy, a0, a1) = apply_model(
            state, batch_images, batch_labels, class_weights
        )
        state = update_model(state, grads)
        epoch_loss.append(loss)

        epoch_accuracy.append(accuracy)
        epoch_a0.append(a0)
        epoch_a1.append(a1)

    train_loss = np.mean(epoch_loss)

    train_accuracy = np.mean(epoch_accuracy)
    a0 = np.mean(epoch_a0)
    a1 = np.mean(epoch_a1)
    return state, train_loss, (train_accuracy, a0, a1)


# def train_epoch(state, train_ds, batch_size, rng):
#     """Train for a single epoch."""
#     batch_images = train_ds[0]
#     batch_labels = train_ds[1]
#     grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
#     state = update_model(state, grads)
#     # epoch_loss.append(loss)
#     # epoch_accuracy.append(accuracy)
#
#     return state, loss, accuracy


def _train_test_split(X, Y, key):
    from jax import random

    key, subkey = random.split(key)

    num_samples = X.shape[0]
    num_test_samples = max(num_samples // 2, 1)

    idxs = random.choice(subkey, num_samples, shape=(num_samples,), replace=False)

    X = X[idxs]  # type: ignore
    Y = Y[idxs]  # type: ignore

    train_X = X[num_test_samples:]
    train_Y = Y[num_test_samples:]

    test_X = X[:num_test_samples]
    test_Y = Y[:num_test_samples]

    return (train_X, train_Y), (test_X, test_Y)


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    from jax import random

    key = random.PRNGKey(0)

    key, subkey = random.split(key)
    X_0 = random.normal(key=subkey, shape=(1000, 2)) - 2

    key, subkey = random.split(key)
    X_1 = random.normal(key=subkey, shape=(1000, 2)) + 2

    X = jnp.concatenate([X_0, X_1], axis=0)
    Y = jnp.concatenate([jnp.zeros((1000,)), jnp.ones((1000,))])

    key, subkey = random.split(key)
    return _train_test_split(X, Y, subkey)


def get_datasets_pyloric() -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
    """Load MNIST train and test datasets into memory."""
    import pickle

    from jax import random

    with open("./pyloric_data_round_0.pkl", "rb") as f:
        theta, X = pickle.load(f)

    is_valid = (X.isnan().sum(axis=1) == 0).float()
    print(f"num valid simulations: {is_valid.sum()}")

    import torch

    theta_mean = torch.mean(theta, dim=0)
    theta_std = 1e-8 + torch.mean((theta - theta_mean) ** 2, dim=0)

    theta = (theta - theta_mean) / theta_std

    theta = jnp.array(theta)
    Y = jnp.array(is_valid)

    key = random.PRNGKey(0)

    from sklearn.model_selection import train_test_split

    theta_train, theta_test, y_train, y_test = train_test_split(
        theta, Y, random_state=42, stratify=Y, train_size=0.5
    )
    return (theta_train, y_train), (theta_test, y_test)
    # return _train_test_split(theta, Y, key)


def create_train_state(rng, X):
    """Creates initial `TrainState`."""
    mlp = CalibrationMLP()
    params = mlp.init(rng, jnp.ones_like(X))["params"]
    tx = optax.adamw(1e-3, weight_decay=1e-1)
    return train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)


def train_and_evaluate():
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    # train_ds, test_ds = get_datasets()
    train_ds, test_ds = get_datasets_pyloric()

    rng = jax.random.PRNGKey(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, train_ds[0])

    batch_size = 10000
    batch_size = min(batch_size, train_ds[0].shape[0])

    max_iter = 1000

    class_weights = jnp.array(
        [1 / (train_ds[1] == 0).sum(), 1 / (train_ds[1] == 1).sum()]
    )
    for epoch in range(1, max_iter):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, (train_accuracy, train_a0, train_a1) = train_epoch(
            state, train_ds, batch_size, input_rng, class_weights
        )
        # print(train_accuracy)
        _, test_loss, (test_accuracy, test_a0, test_a1) = apply_model(
            state, test_ds[0], test_ds[1], class_weights
        )

        if (epoch % max(max_iter // 20, 1)) == 0:
            # logging.info(
            #     "epoch:% 3d, train_loss: %.6f, train_accuracy: %.4f, test_loss: %.6f, test_accuracy: %.4f"
            #     % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
            # )
            logging.info(
                "epoch:% 3d, train_a0: %.4f, train_a1: %.4f, test_a0: %.4f, test_a1: %.4f, test_accuracy: %.4f"
                % (epoch, train_a0 * 100, train_a1 * 100, test_a0 * 100, test_a1 * 100, test_accuracy * 100)
            )

    return state, train_ds, test_ds


if __name__ == "__main__":
    state, train_ds, test_ds = train_and_evaluate()
