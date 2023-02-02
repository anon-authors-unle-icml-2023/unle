import jax
import jax.numpy as jnp
from flax.linen.module import Module, compact

from sbi_ebm.pytypes import Numeric


class StepSizeAjuster(Module):
    momentum: float = 0.9
    min_acceptance_threshold: float = 0.5
    max_acceptance_threshold: float = 0.65
    n_fold = 1.5

    @compact
    def __call__(self, accept_freq: Numeric, step_size: Numeric) -> Numeric:
        is_initialized = self.has_variable("sampling_stats", "avg_accept_freq")

        avg_accept_freq = self.variable(
            "sampling_stats", "avg_accept_freq", lambda x: x, accept_freq
        )

        # update statistics
        if is_initialized:
            avg_accept_freq.value = (
                self.momentum * avg_accept_freq.value
                + (1.0 - self.momentum) * accept_freq
            )

        step_size = jax.lax.cond(
            avg_accept_freq.value < self.min_acceptance_threshold,
            lambda: step_size / self.n_fold,
            lambda: step_size,
        )
        step_size = jax.lax.cond(
            avg_accept_freq.value > self.max_acceptance_threshold,
            lambda: step_size * self.n_fold,
            lambda: step_size,
        )
        return step_size


class LikelihoodMonitor(Module):
    max_iter_no_test_increase: int

    @compact
    def __call__(self, train_likelihood, test_likelihood):
        is_initialized = self.has_variable("likelihood", "max_test_likelihood")

        max_test_likelihood = self.variable(
            "likelihood", "max_test_likelihood", lambda x: x, -1e8
        )

        num_iter_no_test_increase = self.variable(
            "likelihood", "num_iter_no_test_increase", jnp.zeros, ()
        )

        prev_train_likelihood = self.variable(
            "likelihood", "prev_train_likelihood", lambda x: x, -1e8
        )

        # update statistics
        if is_initialized:
            max_test_likelihood.value = jnp.maximum(
                test_likelihood, max_test_likelihood.value
            )

        num_iter_no_test_increase.value = jax.lax.cond(
            jnp.all(
                jnp.array(
                    [
                        test_likelihood < max_test_likelihood.value,
                        train_likelihood > prev_train_likelihood.value,
                    ]
                )
            ),
            lambda x: x + 1,
            lambda x: 0 * x,
            num_iter_no_test_increase.value,
        )
        should_stop = jax.lax.cond(
            num_iter_no_test_increase.value >= self.max_iter_no_test_increase,
            lambda: True,
            lambda: False,
        )
        should_stop = False

        if is_initialized:
            prev_train_likelihood.value = train_likelihood

        # if should_stop:
        #     max_test_likelihood.value = -jnp.inf

        return should_stop
