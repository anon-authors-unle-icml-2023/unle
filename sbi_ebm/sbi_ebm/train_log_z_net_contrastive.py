# See issue #620.
# pytype: disable=wrong-keyword-args
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax import struct
from flax.linen.linear import Dense
from flax.linen.module import Module, compact
from flax.training import train_state
from jax.nn import relu
from sbi_ebm.pytypes import LogDensity_T, Array, PyTreeNode
from sklearn.model_selection import train_test_split

logging.set_verbosity(logging.INFO)


class TrainingConfig(struct.PyTreeNode):
    max_iter: int = struct.field(pytree_node=False, default=100)
    batch_size: int = struct.field(pytree_node=False, default=10000)
    learning_rate: float = struct.field(pytree_node=False, default=1e-2)
    weight_decay: float = struct.field(pytree_node=False, default=1e-4)


class MLPClassifier(Module):
    """A simple MLP model."""

    num_neurons: int = 200

    @compact
    def __call__(
        self, x: Array
    ) -> Array:  # pyright: ignore [reportIncompatibleMethodOverride]
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
        logits = MLPClassifier().apply({"params": params}, images)
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


def create_train_state(rng, X, config: TrainingConfig):
    """Creates initial `TrainState`."""
    mlp = MLPClassifier()
    params = mlp.init(rng, jnp.ones_like(X))["params"]
    tx = optax.adamw(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )
    return train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)


class RatioClassifier(struct.PyTreeNode):
    params: PyTreeNode

    def log_ratio(self, theta, x):
        logits = MLPClassifier().apply(
            {"params": self.params}, jnp.concatenate([theta, x])
        )
        return jax.nn.log_softmax(logits)[..., 1]

class LogZNetContrastive(struct.PyTreeNode):
    params: PyTreeNode

    def predict(self, theta):
        logits = MLPClassifier().apply(
            {"params": self.params}, theta
        )
        return jax.nn.log_softmax(logits)[..., 1] - jax.nn.log_softmax(logits)[..., 0]



def train_classifier(
    params: Array, y: Array, config: TrainingConfig
) -> LogZNetContrastive:
    batch_size = min(config.batch_size, y.shape[0])
    max_iter = config.max_iter

    theta_train, theta_test, y_train, y_test = train_test_split(
        params, y, random_state=43, stratify=y, train_size=0.8
    )
    assert not isinstance(theta_train, list)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, theta_train, config)

    class_weights = jnp.array(
        [1 / (y_train == 0).sum(), 1 / (y_train == 1).sum()]  # type: ignore
    )
    params_trajectory = []
    test_loss_trajectory = []
    max_num_records = 20
    record_every = max(max_iter // max_num_records, 1)

    for epoch in range(1, max_iter):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, (_, _, _) = train_epoch(
            state, (theta_train, y_train), batch_size, input_rng, class_weights
        )
        _, test_loss, (_, _, _) = apply_model(state, theta_test, y_test, class_weights)

        if (epoch % record_every) == 0:
            params_trajectory.append(state.params)
            test_loss_trajectory.append(test_loss)

            logging.info(
                "epoch:% 3d, train_loss: %.6f, test_loss: %.6f"
                % (epoch, train_loss, test_loss)
            )

    step_min_test_loss = jnp.argmin(jnp.array(test_loss_trajectory))
    print(f"min test loss at epoch {step_min_test_loss * record_every}")
    best_params = params_trajectory[step_min_test_loss]

    return LogZNetContrastive(best_params)


class RatioBasedPosteriorLogProb(struct.PyTreeNode):
    prior_log_prob: LogDensity_T
    ratio: RatioClassifier
    x_obs: Array

    @property
    def dim_parameter(self):
        return self.ratio.params["Dense_0"]["kernel"].shape[0] - self.x_obs.shape[0]

    def __call__(self, theta):
        return self.prior_log_prob(theta) + self.ratio.log_ratio(theta, self.x_obs)


class SBIResults(struct.PyTreeNode):
    posterior_log_prob: RatioBasedPosteriorLogProb
    posterior_samples: Array
    x_obs: Array
    num_observation: int
    task_name: str
