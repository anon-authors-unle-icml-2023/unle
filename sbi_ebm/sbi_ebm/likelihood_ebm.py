from time import time
from typing import (Callable, Dict, Literal, NamedTuple,
                    Optional, Tuple, Type, TypeVar, Union, cast)

import cloudpickle
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core.scope import VariableDict
from flax.linen.module import Module
from flax.training import train_state
from jax import grad, jit, random, vmap
from jax._src.flatten_util import ravel_pytree
from jax.random import fold_in
from jax.tree_util import tree_leaves, tree_map
from numpyro import distributions as np_distributions
from typing_extensions import (Self, TypeAlias)

from sbi_ebm.calibration.calibration import CalibrationMLP
from sbi_ebm.data import SBIDataset, SBIParticles
from sbi_ebm.distributions import (DoublyIntractableJointLogDensity, MixedJointLogDensity,
                                   maybe_wrap, maybe_wrap_joint)
from sbi_ebm.neural_networks import MLP, IceBeem
from sbi_ebm.pytypes import (Array, LogDensity_T, Numeric, PRNGKeyArray, PyTreeNode)
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation
from sbi_ebm.samplers.kernels.discrete_gibbs import DiscreteLogDensity
from sbi_ebm.samplers.inference_algorithms.base import InferenceAlgorithm, InferenceAlgorithmFactory, InferenceAlgorithmInfo
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCFactory, SMCParticleApproximation
from sbi_ebm.train_utils import LikelihoodMonitor

# jit = lambda x: x


class ReplayBuffer(struct.PyTreeNode):
    max_size: int
    current_idx: int
    current_size: int
    buffer: Array

    @classmethod
    def create(cls: Type[Self], max_size: int, dim_xs: int) -> Self:
        buffer = jnp.empty((max_size, dim_xs))
        return cls(max_size, 0, 0, buffer)

    def add(self, xs: Array) -> Self:
        num_new_xs = xs.shape[0]
        should_split = num_new_xs + self.current_idx > self.max_size

        if not should_split:
            new_buffer = self.buffer.at[
                self.current_idx : self.current_idx + num_new_xs
            ].set(xs)
        else:
            split_idx = self.max_size - self.current_idx
            new_buffer = self.buffer.at[self.current_idx :].set(xs[:split_idx])
            new_buffer = new_buffer.at[: num_new_xs - split_idx].set(xs[split_idx:])

        new_size = min(self.max_size, self.current_size + num_new_xs)
        new_idx = (self.current_idx + num_new_xs) % self.max_size

        return self.replace(
            buffer=new_buffer, current_size=new_size, current_idx=new_idx
        )


def tree_any(function: Callable[[PyTreeNode], Numeric], tree: PyTreeNode) -> Numeric:
    mapped_tree = tree_map(function, tree)
    return jnp.any(jnp.array(tree_leaves(mapped_tree)))


def _default_base_measure_log_prob(x: Array) -> Array:
    return -0.5 * jnp.sum(jnp.square((x / 10)))
    # return -0.5 * 1e-6 * jnp.sum(jnp.square((x / 4)))  # TODO: more reasaonable value


class EBMLikelihoodConfig(struct.PyTreeNode):
    base_measure_log_prob: Callable[[Array], Array] = struct.field(
        pytree_node=False, default=_default_base_measure_log_prob
    )
    energy_network_type: str = struct.field(pytree_node=False, default="MLP")
    width: int = struct.field(pytree_node=False, default=50)
    depth: int = struct.field(pytree_node=False, default=4)


class OptimizerConfig(NamedTuple):
    learning_rate: float = 0.01
    weight_decay: float = 0.005
    noise_injection_val: float = 0.02


class LikelihoodEstimationConfig(struct.PyTreeNode):
    enabled: bool = struct.field(pytree_node=False)
    num_particles: int = struct.field(pytree_node=False)
    alg: SMCFactory = struct.field(pytree_node=True)
    use_warm_start: bool = struct.field(pytree_node=False)
    init_dist: np_distributions.Distribution = struct.field(pytree_node=False)


# Note that TypeVar constraints are not transparent w.r.t constraints subtypes.
EBM_MODELS_TYPES_T: TypeAlias = Literal["joint_tilted", "joint_unbiased", " joint_tilted_discrete", "likelihood", "ratio"]


# XXX: necessary to put the Generic class first to avoid
# ``type object 'TrainingConfig' has no attribute '__parameters__'``
class TrainingConfig(struct.PyTreeNode):
    optimizer: OptimizerConfig
    ebm: EBMLikelihoodConfig
    sampling_cfg_first_iter: InferenceAlgorithmFactory
    sampling_cfg: InferenceAlgorithmFactory
    # marked as non pytree node because it is futher used as non-pytree node, within
    # wrappers - an inconsistency between input-output node type leads to leaked jax
    # tracers.
    sampling_init_dist: Union[
        Literal["data"], np_distributions.Distribution
    ] = struct.field(pytree_node=False)
    likelihood_estimation_config: LikelihoodEstimationConfig = struct.field(
        pytree_node=True
    )
    max_iter: int = struct.field(pytree_node=False)
    num_particles: int = struct.field(pytree_node=False, default=1000)
    use_warm_start: bool = struct.field(pytree_node=False, default=False)
    verbose: bool = struct.field(pytree_node=False, default=True)
    patience: int = struct.field(pytree_node=False, default=1000)
    max_num_recordings: int = struct.field(pytree_node=False, default=100)
    gradient_penalty_val: float = struct.field(pytree_node=True, default=0)
    batch_size: Optional[int] = struct.field(pytree_node=False, default=None)
    restart_every: Optional[int] = struct.field(pytree_node=False, default=None)
    batching_enabled: bool = struct.field(pytree_node=False, default=False)
    recording_enabled: bool = struct.field(pytree_node=False, default=False)
    checkpoint_every: Optional[int] = struct.field(pytree_node=False, default=None)
    n_iter_warmup: int = struct.field(pytree_node=False, default=5000)
    select_based_on_test_loss: bool = struct.field(pytree_node=False, default=True)
    ebm_model_type: EBM_MODELS_TYPES_T = struct.field(pytree_node=False, default="joint_tilted")
    update_all_particles: bool = struct.field(pytree_node=False, default=True)
    concat_all_chain_iterations: bool = struct.field(pytree_node=False, default=False)
    estimate_log_normalizer: bool = struct.field(pytree_node=False, default=False)


# XXX: ditto
class TrainStateMixin(struct.PyTreeNode):
    training_algs: Tuple[InferenceAlgorithm, ...]
    log_Z_algs: Tuple[SMC, ...]
    loss: VariableDict
    has_nan: bool
    has_converged: bool
    replay_buffer: Optional[ReplayBuffer]
    opt_is_diverging: bool = False
    calibration_net: Optional[CalibrationMLP] = None


class TrainState(TrainStateMixin, train_state.TrainState):
    # XXX: tx is overridden in an incompatible manner and breaks subtype polymorphism
    tx: Tuple[optax.GradientTransformation, ...] = struct.field(pytree_node=False)  # pyright: ignore[reportIncompatibleVariableOverride]
    opt_state: Tuple[optax.OptState, ...]


class MiniTrainState(struct.PyTreeNode):
    params: PyTreeNode


class TrainingStats(struct.PyTreeNode):
    loss: Dict
    sampling: Optional[InferenceAlgorithmInfo]
    mmd: Numeric = 0
    grad_norm: Numeric = 0


class Energy(Module):
    energy_network_type: str
    width: int = 50
    depth: int = 4

    def setup(self):
        if self.energy_network_type == "MLP":
            self.energy_network = MLP(width=self.width, depth=self.depth)
        elif self.energy_network_type == "IceBeem":
            self.energy_network = IceBeem()
        else:
            raise ValueError

    def __call__(self, inputs: Tuple[Array, Array]):

        if self.energy_network_type == "MLP":
            (z, x) = inputs
            zx = jnp.concatenate([z, x])
            ret = self.energy_network(zx)
            return ret[0]  # type: ignore
        elif self.energy_network_type == "IceBeem":
            ret = self.energy_network(inputs)
            return ret
        else:
            raise ValueError


def energy(energy_network_type, width: int = 50, depth: int = 4) -> Energy:
    # using convention of:
    # https://github.com/google/flax/blob/main/examples/vae/train.py EBM instances are
    # stateless - you can literally create a new one each time you want to call an EBM
    # method - this this what this function underlines.
    return Energy(energy_network_type, width, depth)


class _EBMLikelihoodLogDensity(struct.PyTreeNode):
    """
    An unnormalized, likelihood function parametrized by an energy function.
    """

    params: PyTreeNode
    config: EBMLikelihoodConfig

    def __call__(self, param: Array, obs: Array) -> Numeric:
        ret = (
            -energy(self.config.energy_network_type, self.config.width, self.config.depth).apply(self.params, (param, obs))  # type: ignore
            + self.config.base_measure_log_prob(obs)
            + self.config.base_measure_log_prob(param)  # for regularization purposes
            # - 1e80 * (jnp.any(jnp.abs(obs) > 50))
        )
        return ret


class _EBMJointLogDensity(DoublyIntractableJointLogDensity):
    """
    A differentiable, doubly intractable joint model that is sampled from during training.
    """

    log_likelihood: _EBMLikelihoodLogDensity  # pyright: ignore[reportIncompatibleMethodOverride]
    dim_param: int = struct.field(pytree_node=False)

    def set_params(self, params: PyTreeNode):
        return self.replace(log_likelihood=self.log_likelihood.replace(params=params))

    def tilted_log_joint(self, x: Array) -> Numeric:
        return super().tilted_log_joint(x)



class _EBMMixedJointLogDensity(MixedJointLogDensity):
    log_likelihood: _EBMLikelihoodLogDensity  # pyright: ignore[reportIncompatibleMethodOverride]
    dim_param: int = struct.field(pytree_node=False)

    def set_params(self, params: PyTreeNode):
        return self.replace(log_likelihood=self.log_likelihood.replace(params=params))

    def tilted_log_joint(self, x: Array) -> Numeric:
        return super().tilted_log_joint(x)


class _EBMLikelihood(np_distributions.Distribution):
    # https://github.com/pyro-ppl/numpyro/issues/1317
    # Distributions instances cannot be vmapped, and thus are not used during the
    # training loop. Instances of this class are return by sbi_ebm at the end of
    # training for ease of potential integration downstream numpyro applications.
    arg_contraints = {"params": None, "config": None, "z": None}

    def __init__(self, params: PyTreeNode, config: EBMLikelihoodConfig, param: Array):
        self.params = params
        self.config = config
        self.param = param

        if self.config.energy_network_type == "MLP":
            self._dim_x = (
                params["params"]["energy_network"]["layers_0"]["kernel"].shape[0]
                - param.shape[0]
            )
        else:
            self._dim_x = params["params"]["energy_network"]["x_net_0"]["kernel"].shape[
                0
            ]

        super(_EBMLikelihood, self).__init__(batch_shape=(), event_shape=(self._dim_x,))

    def log_prob(self, x):
        return (
            -energy(self.config.energy_network_type, self.config.width, self.config.depth).apply(  # type: ignore
                self.params, (self.param, x)
            )
            + self.config.base_measure_log_prob(x)
            + self.config.base_measure_log_prob(self.param)
        )  # for regularization purposes


class LikelihoodFactory(struct.PyTreeNode):
    params: PyTreeNode
    config: EBMLikelihoodConfig
    is_doubly_intractable: bool = struct.field(pytree_node=False, default=False)

    def __call__(self, param: Array) -> _EBMLikelihood:
        return _EBMLikelihood(params=self.params, config=self.config, param=param)




class _EBMRatio(struct.PyTreeNode):
    params: PyTreeNode
    config: EBMLikelihoodConfig

    def __call__(self, param: Array, x: Array):
        return (
            -energy(self.config.energy_network_type, self.config.width, self.config.depth).apply(  # type: ignore
                self.params, (param, x)
            )
            # + self.config.base_measure_log_prob(x)
            # + self.config.base_measure_log_prob(param)
        )  # for regularization purposes


class _EBMDiscreteJointDensity(DiscreteLogDensity):
    ratio: _EBMRatio = struct.field(pytree_node=True, default=None)
    calibration_net: Optional[CalibrationMLP] = None

    def log_prob(self, theta:  Array, x: Array) -> Numeric:
        if self.calibration_net is None:
            return self.ratio(theta, x)
        else:
            return self.ratio(theta, x) - self.calibration_net.log_prob(theta)


    def set_params(self, params: PyTreeNode):
        return self.replace(ratio=self.ratio.replace(params=params))


def _print_consumer(
    arg: Tuple[TrainState, TrainingConfig, TrainingStats], _
):
    state, config, stats = arg

    _iter_str = f"{state.step}/{config.max_iter}"

    # log_Z_str = f"{state.log_Z_init[0]._Z_log_space:.3e}"
    log_Z_str = f"0"

    if state.has_converged:
        print(
            f"iteration {_iter_str:<10}: Validation likelihood stopped "
            f"increasing, early stopping the algorithm.",
            # flush=True,
        )
    elif state.has_nan:
        print(f"iteration {_iter_str:<10}: algorithm encountered nans", flush=True)
    else:
        print(
            f"iteration {_iter_str:<10}: {stats.grad_norm:<10.3f}"
            f"unnormalized_train_log_l={stats.loss['unnormalized_train_log_l']:<10.3f} "
            f"unnormalized_test_log_l={stats.loss['unnormalized_test_log_l']:<10.3f} "
            f"train_log_l={stats.loss['train_log_l']:<10.3f}"
            f"test_log_l={stats.loss['test_log_l']:<10.3f}"
            f"ebm_log_l={stats.loss['ebm_samples_train_log_l']:<10.3f}"
            # f"step_size={state.mala_step_size:<8.4f} "
            # f"has_converged={state.has_converged:<3} "
            # f"has_nan={state.has_nan:<3} "
            f"log_Z={log_Z_str}",
            flush=True,
        )


def maybe_print_info(
    state: TrainState, config: TrainingConfig, stats: TrainingStats
):

    should_print = jnp.any(
        ravel_pytree(
            [
                state.step % max(config.max_iter // 20, 1) == 0,
                state.has_converged,
                state.has_nan,
            ]
        )[0]
    )

    if should_print:
        _print_consumer((state, config, stats), None)
    # _ = jax.lax.cond(
    #     should_print,
    #     lambda arg: host_callback.id_tap(_print_consumer, arg),
    #     lambda arg: arg,
    #     (state, config, stats),
    # )
    return state.step


class TrainerResults(struct.PyTreeNode):
    init_state: TrainState
    final_state: TrainState
    best_state: TrainState
    trajectory: Optional[TrainState]
    stats: Optional[TrainingStats]
    datasets: Optional[Tuple[SBIDataset, ...]]
    config: TrainingConfig
    ratio: Optional[_EBMRatio] = None
    likelihood_factory: Optional[LikelihoodFactory] = None
    time: float = 0.0



LD_T = TypeVar("LD_T", DoublyIntractableJointLogDensity, DiscreteLogDensity)


uniform_log_density = maybe_wrap_joint(lambda theta, x: 1.)  # pyright: ignore


class Trainer:
    @staticmethod
    def pseudo_loss(params, ebm_samples: ParticleApproximation, true_samples: SBIParticles, ebm_config: EBMLikelihoodConfig, noise_injection_val: float, key: PRNGKeyArray):
        dim_z = true_samples.dim_params
        def energy_fn(z, x):
            # the total "energy" of the joint is (minus) the joint log-probability.
            # However, the prior and base measure do not depend on the neural
            # network: the only remaining term that actually has a gradient is the
            # energy network itself.
            return energy(ebm_config.energy_network_type, ebm_config.width, ebm_config.depth).apply(
                params, (z, x)
            )

        noise: Array = noise_injection_val * random.normal(
            key, true_samples.xs.shape
        )

        energy_true_samples = jnp.average(
            vmap(energy_fn)(
                true_samples.params,  # + noise[:, :dim_z],
                true_samples.observations + noise[:, dim_z:],
            ),
            weights=true_samples.normalized_ws,
        )

        energy_ebm_samples = jnp.average(
            vmap(energy_fn)(ebm_samples.xs[:, :dim_z], ebm_samples.xs[:, dim_z:]),
            weights=ebm_samples.normalized_ws,
        )

        return (
            energy_true_samples
            - energy_ebm_samples
            # + 0.01 * (energy_ebm_samples ** 2 + energy_ebm_samples ** 2)
            # + logsumexp(energy_ebm_samples) + logsumexp(-energy_ebm_samples)
            # + 100 * (jax.nn.logsumexp(energy_ebm_samples) + jax.nn.logsumexp(-energy_ebm_samples))
        )



    def compute_ebm_approx(self, alg: InferenceAlgorithm, log_joint: Union[_EBMDiscreteJointDensity, _EBMJointLogDensity], params: PyTreeNode, key: PRNGKeyArray, true_samples: SBIParticles) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        alg = alg.set_log_prob(log_joint.set_params(params=params))
        # call the class method to prevent spurious recompilations.
        key, subkey = random.split(key)
        alg, results = type(alg).run_and_update_init(alg, subkey)
        return alg, results.samples

    def compute_normalized_ebm_approx(self, likelihood_estimation_alg: SMC, log_joint: Union[_EBMDiscreteJointDensity, _EBMJointLogDensity], params: PyTreeNode, key: PRNGKeyArray) -> Tuple[SMC, SMCParticleApproximation]:
        log_joint = log_joint.set_params(params)
        likelihood_estimation_alg = likelihood_estimation_alg.set_log_prob(log_joint)

        key, subkey = random.split(key)
        likelihood_estimation_alg, log_Z_results = type(likelihood_estimation_alg).run_and_update_init(likelihood_estimation_alg, subkey)
        return likelihood_estimation_alg, log_Z_results.samples


    def estimate_log_likelihood_gradient(
        self,
        params: PyTreeNode,
        true_samples: SBIParticles,
        ebm_samples: ParticleApproximation,
        ebm_config: EBMLikelihoodConfig,
        noise_injection_val: float,
        key: PRNGKeyArray,
        log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity],
    ) -> PyTreeNode:
        objective_gradient = grad(self.pseudo_loss)(params, ebm_samples, true_samples, ebm_config, noise_injection_val, key)
        return objective_gradient

    def estimate_train_and_val_loss(
        self,
        params: PyTreeNode,
        dataset: SBIDataset,
        ebm_samples: ParticleApproximation,
        ebm_samples_log_Z: SMCParticleApproximation,
        key: PRNGKeyArray,
        log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity],
    ) -> Dict[str, Numeric]:
        log_joint = log_joint.set_params(params)

        avg_unnormalized_train_log_prob = jnp.average(
            vmap(log_joint)(dataset.train_samples.xs),
            weights=dataset.train_samples.normalized_ws,
        )

        avg_unnormalized_test_log_prob = jnp.average(
            vmap(log_joint)(dataset.test_samples.xs),
            weights=dataset.test_samples.normalized_ws,
        )

        avg_unnormalized_ebm_samples_log_prob = jnp.average(
            vmap(log_joint)(ebm_samples.xs), weights=ebm_samples.normalized_ws
        )

        avg_unnormalized_train_log_prob = cast(Numeric, avg_unnormalized_train_log_prob)
        avg_unnormalized_test_log_prob = cast(Numeric, avg_unnormalized_test_log_prob)
        avg_unnormalized_ebm_samples_log_prob = cast(
            Numeric, avg_unnormalized_ebm_samples_log_prob
        )

        likelihood_vals = {
            "unnormalized_train_log_l": avg_unnormalized_train_log_prob,
            "unnormalized_test_log_l": avg_unnormalized_test_log_prob,
            "ebm_samples_train_log_l": 0.0,
            "train_log_l": 0.0,
            "test_log_l": 0.0,
        }

        likelihood_vals["train_log_l"] = (
            avg_unnormalized_train_log_prob - ebm_samples_log_Z._Z_log_space
        )
        likelihood_vals["test_log_l"] = (
            avg_unnormalized_test_log_prob - ebm_samples_log_Z._Z_log_space
        )
        likelihood_vals["ebm_samples_train_log_l"] = (
            avg_unnormalized_ebm_samples_log_prob - ebm_samples_log_Z._Z_log_space
        )

        return likelihood_vals

    def estimate_value_and_grad(
        self,
        params: PyTreeNode,
        ebm_config: EBMLikelihoodConfig,
        noise_injection_val: float,
        proposal_log_prob: LogDensity_T,
        ebm_samples: ParticleApproximation,
        ebm_samples_log_Z: Optional[SMCParticleApproximation],
        likelihood_estimation_config: LikelihoodEstimationConfig,
        key: PRNGKeyArray,
        dataset: SBIDataset,
        ebm_model_type: str,
        use_warm_start: bool,
        num_particles: int,
        step: int,
        log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity],
    ) -> Tuple[TrainingStats, PyTreeNode]:

        key, subkey = random.split(key)
        grads = self.estimate_log_likelihood_gradient(
            params,
            dataset.train_samples,
            ebm_samples,
            ebm_config,
            noise_injection_val,
            subkey,
            log_joint,
        )

        if likelihood_estimation_config.enabled:
            key, subkey = random.split(key)
            assert isinstance(log_joint, _EBMJointLogDensity)
            assert ebm_samples_log_Z is not None
            loss_dict = self.estimate_train_and_val_loss(
                params,
                dataset,
                ebm_samples,
                ebm_samples_log_Z,
                subkey,
                log_joint,
            )
            stats = TrainingStats(
                loss=loss_dict,
                # sampling=tree_map(jnp.mean, training_results.info),
                sampling=None,
                grad_norm=jnp.sum(jnp.square(ravel_pytree(grads)[0]))
            )
            return grads, stats
        else:
            _keys = (
                "unnormalized_train_log_l",
                "unnormalized_test_log_l",
                "train_log_l",
                "test_log_l",
                "ebm_samples_train_log_l",
            )
            stats = TrainingStats(
                # loss={k: 0.0 for k in _keys}, sampling=tree_map(jnp.mean, training_results.info),
                loss={k: 0.0 for k in _keys}, sampling=None,
                grad_norm=jnp.sum(jnp.square(ravel_pytree(grads)[0]))
            )
            if ebm_model_type == "ratio":
                stats.loss['unnormalized_train_log_l'] = jnp.average(
                    vmap(log_joint.set_params(params))((dataset.train_samples.indices, dataset.train_samples.indices)),
                    weights=dataset.train_samples.normalized_ws,
                )
            elif ebm_model_type == "likelihood":
                assert isinstance(log_joint, _EBMJointLogDensity)
                stats.loss['unnormalized_train_log_l'] = jnp.average(
                    vmap(log_joint.log_likelihood.replace(params=params))(dataset.train_samples.params, dataset.train_samples.observations),
                    weights=dataset.train_samples.normalized_ws,
                )

                stats.loss['unnormalized_test_log_l'] = jnp.average(
                    vmap(log_joint.log_likelihood.replace(params=params))(dataset.test_samples.params, dataset.test_samples.observations),
                    weights=dataset.test_samples.normalized_ws,
                )

                # stats.loss['ebm_samples_train_log_l'] = jnp.average(
                #     vmap(log_joint.log_likelihood)(ebm_samples.xs[:, :dataset.dim_params], ebm_samples.xs[:, dataset.dim_params:]),
                #     weights=ebm_samples.normalized_ws
                # )
                stats.loss['ebm_samples_train_log_l'] = 0.
            else:
                stats.loss['unnormalized_train_log_l'] = jnp.average(
                    vmap(log_joint.set_params(params))(dataset.train_samples.xs),
                    weights=dataset.train_samples.normalized_ws,
                )
                stats.loss['unnormalized_test_log_l'] = jnp.average(
                    vmap(log_joint.set_params(params))(dataset.test_samples.xs),
                    weights=dataset.test_samples.normalized_ws,
                )
            # __import__('pdb').set_trace()
            return grads, stats


    def _resolve_proposal_distribution(self, config: TrainingConfig, datasets: Tuple[SBIDataset]) -> Tuple[np_distributions.Distribution]:
        if config.ebm_model_type == "ratio":
            log_density_uniform = np_distributions.DiscreteUniform(low=jnp.zeros((2,)), high=len(datasets[0].train_samples.observations) * jnp.ones((2,))).to_event()
            return tuple(log_density_uniform for _ in range(len(datasets)))
        else:
            assert isinstance(config.sampling_init_dist, np_distributions.Distribution)
            return tuple(config.sampling_init_dist for _ in range(len(datasets)))

    def _resolve_proposal_particles(self, config: TrainingConfig, datasets: Tuple[SBIDataset], key: PRNGKeyArray) -> Tuple[Array]:
        key, key_zx0s = random.split(key)
        keys_zx0s = random.split(key_zx0s, num=len(datasets))

        indexes = tuple(
            random.choice(k, d.train_samples.num_samples, (config.num_particles,))
            for k, d in zip(keys_zx0s, datasets)
        )
        return tuple(datasets[i].train_samples.xs[idxs] for i, idxs in enumerate(indexes))


    def _init_training_alg(
        self,
        config: TrainingConfig,
        datasets: Tuple[SBIDataset, ...],
        params: PyTreeNode,
        key: PRNGKeyArray,
        log_joints: Tuple[Union[_EBMJointLogDensity, _EBMDiscreteJointDensity, _EBMMixedJointLogDensity]],
        use_first_iter_cfg: bool = False,
        # algs: Optional[Tuple[InferenceAlgorithm]] = None,
    ) -> Tuple[InferenceAlgorithm, ...]:
        assert config.num_particles is not None  # type narrowing
        assert config.sampling_init_dist is not None  # type narrowing

        if use_first_iter_cfg:
            algs = tuple([config.sampling_cfg_first_iter.build_algorithm(log_prob=log_joint.set_params(params)) for log_joint in log_joints])
        else:
            algs = tuple([config.sampling_cfg.build_algorithm(log_prob=log_joint.set_params(params)) for log_joint in log_joints])

        if isinstance(config.sampling_init_dist, np_distributions.Distribution):
            dists = self._resolve_proposal_distribution(config, datasets)
            key, subkey = random.split(key)
            algs = tuple(a.init(fold_in(subkey, i), d) for i, (a, d) in enumerate(zip(algs, dists)))
        else:
            particles = self._resolve_proposal_particles(config, datasets, key)
            assert isinstance(algs[0], MCMCAlgorithm)
            algs = tuple(alg.init_from_particles(ps) for alg, ps in zip(algs, particles))  # type: ignore
        return algs

    def _init_log_Z_alg(
        self,
        config: TrainingConfig,
        datasets: Tuple[SBIDataset, ...],
        key: PRNGKeyArray,
        log_joints: Tuple[Union[_EBMJointLogDensity, _EBMDiscreteJointDensity, _EBMMixedJointLogDensity]],
    ) -> Tuple[SMC, ...]:
        algs = tuple(config.likelihood_estimation_config.alg.build_algorithm(log_prob=log_joint) for log_joint in log_joints)
        dists = (config.likelihood_estimation_config.init_dist,) * len(datasets)
        key, subkey = random.split(key)
        if config.likelihood_estimation_config.enabled:
            return tuple(alg.init(fold_in(subkey, i), d) for i, (alg, d) in enumerate(zip(algs, dists)))
        else:
            return algs

    def initialize_state(
        self,
        datasets: Tuple[SBIDataset, ...],
        config: TrainingConfig,
        key: PRNGKeyArray,
        calibration_net: Optional[CalibrationMLP] = None,
        use_first_iter_cfg: bool = False,
        params: Optional[PyTreeNode] = None,
    ) -> TrainState:
        # 1. EBM
        key, key_model = random.split(key, 2)
        _z, _x = jnp.ones((datasets[0].dim_params,)), jnp.ones(
            (datasets[0].dim_observations,)
        )
        if params is None:
            params = energy(config.ebm.energy_network_type, config.ebm.width, config.ebm.depth).init(
                key_model, (_z, _x)
            )

        # 2. OPTIMIZER
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            # init_value=config.optimizer.learning_rate,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=max(0, min(50, config.max_iter // 2)),
            decay_steps=config.max_iter,
            # end_value=config.optimizer.learning_rate
            end_value=config.optimizer.learning_rate / 50,
        )
        txs = tuple([optax.chain(
            #  optax.clip_by_global_norm(100.),
            optax.clip(5.0),
            optax.adamw(
                learning_rate=schedule_fn, weight_decay=config.optimizer.weight_decay
            ),
            # optax.sgd(learning_rate=schedule_fn)
        )]* len(datasets))
        opt_state = tuple(tx.init(params) for tx in txs)

        log_joints = self._make_log_joint(params, config, datasets, calibration_net)

        # 3.a PARTICLE APPROXIMATION (gradient)
        key, key_init_particles = random.split(key)
        training_algs = self._init_training_alg(config, datasets, params, key_init_particles, log_joints, use_first_iter_cfg)

        # 3.b PARTICLE APPROXIMATION (log Z)
        key, key_init_log_Z_particles = random.split(key)
        log_Z_algs = self._init_log_Z_alg(
            config, datasets, key_init_log_Z_particles, log_joints
        )

        # 4. LIKELIHOOD MONITOR
        key, key_loss_monitor = random.split(key)
        loss_state = LikelihoodMonitor(config.patience).init(key_loss_monitor, 0.0, 0.0)

        assert config.num_particles is not None
        state = TrainState(
            apply_fn=energy(config.ebm.energy_network_type, config.ebm.width, config.ebm.depth).apply,
            tx=txs,
            params=params,
            opt_state=opt_state,
            step=0,
            training_algs=training_algs,
            log_Z_algs=log_Z_algs,
            loss=loss_state,
            has_nan=False,
            has_converged=False,
            # replay_buffer=ReplayBuffer.create(100000, len(datasets[0].train_samples.observations)),
            replay_buffer=None,
            opt_is_diverging=False,
        )
        return state

    def get_batches(
        self, datasets: Tuple[SBIDataset], batch_size: Optional[int], key: PRNGKeyArray
    ) -> Tuple[SBIDataset]:
        batched_datasets = []
        if batch_size is not None:
            for d in datasets:
                # XXX: important to randomize even when batch_size >= d.train_samples.num_samples
                # due to non-randomization of the batch indices for ebm particles in subselection
                # happening later. TODO: fix this
                batch_size = min(batch_size, d.train_samples.num_samples)
                key, subkey = random.split(key)
                idxs = random.choice(
                    subkey, d.train_samples.xs.shape[0], shape=(batch_size,)
                )
                batched_dataset = d._replace(
                    train_samples=tree_map(
                        lambda x: x[idxs], d.train_samples
                    )
                )
                batched_datasets.append(batched_dataset)
            return tuple(batched_datasets)
        else:
            return datasets

    def train_step(
        self,
        state: TrainState,
        datasets: Tuple[SBIDataset, ...],
        config: TrainingConfig,
        key: PRNGKeyArray,
        entire_datasets: Tuple[SBIDataset, ...],
    ) -> Tuple[TrainState, Tuple[TrainState, TrainingStats]]:
        # print('jitting!')

        # compute 𝛁 KL (p_data, p_ebm)
        all_grads = []
        all_training_algs = []
        all_likelihood_estimation_algs = []
        all_stats = []
        all_updates = []
        all_opt_states = []

        # print(len(datasets))
        # if config.update_all_particles or len(datasets) == 1:

        if config.update_all_particles or True:
            # print(len(range(len(datasets))), len(state.tx), len(state.opt_state), len(self.log_joints), len(datasets), len(state.training_algs), len(state.log_Z_algs))
            i = 0
            for tx, opt_state, log_joint, dataset, alg, l_alg in zip(state.tx, state.opt_state, self.log_joints, datasets, state.training_algs, state.log_Z_algs):
                key, subkey = random.split(key)
                training_alg, results = self.compute_ebm_approx(alg, log_joint, state.params, subkey, dataset.train_samples)

                if config.likelihood_estimation_config.enabled:
                    key, subkey = random.split(key)
                    likelihood_estimation_alg, log_Z_results = self.compute_normalized_ebm_approx(l_alg, log_joint, state.params, subkey)
                else:
                    likelihood_estimation_alg, log_Z_results = l_alg, None

                key, subkey = random.split(key)
                grads, stats = self.estimate_value_and_grad(
                    params=state.params,
                    ebm_config=config.ebm,
                    noise_injection_val=config.optimizer.noise_injection_val,
                    proposal_log_prob=maybe_wrap(dataset.prior.log_prob),  # type: ignore
                    ebm_samples=results,
                    ebm_samples_log_Z=log_Z_results,
                    likelihood_estimation_config=config.likelihood_estimation_config,
                    key=subkey,
                    dataset=dataset,
                    ebm_model_type=config.ebm_model_type,
                    use_warm_start=config.use_warm_start,
                    num_particles=config.num_particles,
                    step=state.step,
                    log_joint=log_joint
                )

                updates, opt_state = tx.update(
                    grads, opt_state, params=state.params,
                )

                all_grads.append(grads)
                all_training_algs.append(training_alg)
                all_likelihood_estimation_algs.append(likelihood_estimation_alg)
                all_stats.append(stats)
                all_updates.append(updates)
                all_opt_states.append(opt_state)


            grads = tree_map(lambda *args: 1 / len(datasets) * sum(args), *all_grads)
            stats = tree_map(lambda *args: 1 / len(datasets) * sum(args), *all_stats)
            updates = tree_map(lambda *args: 1 / len(datasets) * sum(args), *all_updates)
        else:
            key, subkey = random.split(key)
            # idx = random.randint(subkey, shape=(), minval=0, maxval=len(datasets))
            idx = state.step % len(datasets)

            zero_grads = tree_map(lambda x: 0. * x, state.params)
            loss_stats = {k: 0. for k in ("unnormalized_train_log_l", "unnormalized_test_log_l", "train_log_l", "test_log_l", "ebm_samples_train_log_l")}
            zero_stats = TrainingStats(loss=loss_stats, sampling=None, grad_norm=jnp.sum(jnp.square(ravel_pytree(zero_grads)[0])))
            zero_update = tree_map(lambda x: 0. * x, state.tx[0].update(zero_grads, state.opt_state[0], params=state.params)[0])

            grads = zero_grads
            stats = zero_stats
            updates = zero_update

            for _i, tx, opt_state, log_joint, dataset, alg, l_alg in zip(range(len(datasets)), state.tx, state.opt_state, self.log_joints, datasets, state.training_algs, state.log_Z_algs):
                key, subkey = random.split(key)
                training_alg, results = jax.lax.cond(_i == idx, lambda: self.compute_ebm_approx(alg, log_joint, state.params, subkey, dataset.train_samples), lambda: (alg, alg._init_state))

                if config.likelihood_estimation_config.enabled:
                    key, subkey = random.split(key)
                    likelihood_estimation_alg, log_Z_results = jax.lax.cond(_i == idx, lambda: self.compute_normalized_ebm_approx(l_alg, log_joint, state.params, subkey), lambda: (l_alg, l_alg._init_state))
                else:
                    likelihood_estimation_alg, log_Z_results = l_alg, None

                key, subkey = random.split(key)
                this_grads, this_stats = self.estimate_value_and_grad(
                    params=state.params, ebm_config=config.ebm, noise_injection_val=config.optimizer.noise_injection_val, proposal_log_prob=maybe_wrap(dataset.prior.log_prob), ebm_samples=results, ebm_samples_log_Z=log_Z_results, likelihood_estimation_config=config.likelihood_estimation_config,
                    key=subkey, dataset=dataset, ebm_model_type=config.ebm_model_type, use_warm_start=config.use_warm_start, num_particles=config.num_particles, step=state.step, log_joint=log_joint
                )

                this_update, this_opt_state = tx.update(this_grads, opt_state, params=state.params)
                opt_state = jax.lax.cond(_i == idx, lambda: this_opt_state, lambda: opt_state)

                grads = tree_map(lambda x, y: x + (idx == _i) * y, grads, this_grads)
                stats = tree_map(lambda x, y: x + y/len(datasets), stats, this_stats)
                updates = tree_map(lambda x, y: x + (idx == _i) * y, updates, this_update)

                all_training_algs.append(training_alg)
                all_likelihood_estimation_algs.append(likelihood_estimation_alg)
                all_opt_states.append(opt_state)

        all_training_algs_tuple = tuple(all_training_algs)
        all_logZ_algs_tuple = tuple(all_likelihood_estimation_algs)

        # update EBM parameters
        t0 = time()
        params = optax.apply_updates(state.params, updates)

        # update train and test moving averages
        _, loss_monitor_state = LikelihoodMonitor(config.patience).apply(
            state.loss,
            stats.loss.get("train_log_l", 0.0),
            stats.loss.get("test_log_l", 0.0),
            mutable=list(state.loss.keys()),
        )

        # update state
        if not config.use_warm_start:
            # reinitialize previous sampler state using new particles
            key, key_particles = random.split(key)
            all_training_algs_tuple = self._init_training_alg(
                config, entire_datasets, params, key_particles, self.log_joints
            )

        if (
            config.likelihood_estimation_config.enabled
            and not config.likelihood_estimation_config.use_warm_start
        ):
            key, key_log_Z_particles = random.split(key)
            all_logZ_algs_tuple = self._init_log_Z_alg(
                config, datasets, key_log_Z_particles, self.log_joints
            )

        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=tuple(all_opt_states),
            loss=loss_monitor_state,
            training_algs=all_training_algs_tuple,
            log_Z_algs=all_logZ_algs_tuple,
            has_converged=False,
        )
        has_nan = tree_any(lambda x: jnp.any(jnp.isnan(x)), new_state)

        sum_grad_norms = jnp.sum(jnp.square(ravel_pytree(grads)[0]))
        opt_is_diverging = sum_grad_norms > 1e8

        new_state = new_state.replace(
            has_nan=has_nan, opt_is_diverging=opt_is_diverging
        )
        # print('update opt time', time() - t0)

        # _ = maybe_print_info(new_state, config, stats)
        return new_state, (new_state, stats)


    def _make_log_joint(self, params: PyTreeNode, config: TrainingConfig, datasets: Tuple[SBIDataset], calibration_net: Optional[CalibrationMLP]) -> Tuple[Union[_EBMDiscreteJointDensity, _EBMJointLogDensity, _EBMMixedJointLogDensity]]:

        self.ebm_likelihood_log_density = _EBMLikelihoodLogDensity(
            params,
            config.ebm,
        )

        if config.ebm_model_type == "ratio":
            assert len(datasets) == 1
            _log_ratio = _EBMRatio(params=params, config=config.ebm)
            # ratio is proportional to joint since joint is uniform over training samples.
            self.log_joints = tuple([_EBMDiscreteJointDensity(
                _thetas=d.train_samples.params,
                # XXX: stop overloading x/xs
                _xs=d.train_samples.observations,
                ratio=_log_ratio,
            ) for d in datasets])
        elif config.ebm_model_type == "joint_tilted_discrete":
            self.log_joints = tuple([_EBMMixedJointLogDensity(
                # log_prior=maybe_wrap(lambda x: datasets[0].prior.log_prob(x)),
                log_prior=maybe_wrap(d.prior.log_prob),  # type: ignore
                log_likelihood=self.ebm_likelihood_log_density,
                dim_param=d.dim_params,
                thetas=d.train_samples.params,
            ) for d in datasets])
        else:
            self.log_joints = tuple([_EBMJointLogDensity(
                # log_prior=maybe_wrap(lambda x: datasets[0].prior.log_prob(x)),
                log_prior=maybe_wrap(d.prior.log_prob),  # type: ignore
                log_likelihood=self.ebm_likelihood_log_density,
                dim_param=d.dim_params,
            ) for d in datasets])

        return self.log_joints

    def train_ebm_likelihood_model(
        self,
        datasets: Tuple[SBIDataset, ...],
        config: TrainingConfig,
        key: PRNGKeyArray,
        init_state: Optional[TrainState] = None,
    ) -> TrainerResults:
        t0_init_training = time()
        print("number of datasets: ", len(datasets))
        self.num_particles = config.num_particles

        if init_state is None:
            key, key_init = random.split(key)
            init_state = self.initialize_state(datasets, config, key_init, use_first_iter_cfg=True)


        best_state = init_state

        jitted_train_step = jit(self.train_step)
        # jitted_train_step = self.train_step

        import queue
        import collections
        n = 10
        record_stable_state_every = 20
        last_n_stable_states = collections.deque(maxlen=n)

        outputs = []
        key, subkey = random.split(key)
        batched_datasets = self.get_batches(datasets, config.batch_size, subkey)
        key, subkey = random.split(key)

        t0 = time()
        print('first step...')
        state, output = jitted_train_step(
            init_state,
            batched_datasets,
            config.replace(update_all_particles=True),
            subkey,
            datasets,
        )
        print('....done.')
        state = cast(TrainState, state)
        # print('train step time', time() - t0)

        # __import__('pdb').set_trace()
        key, subkey = random.split(key)
        training_algs = self._init_training_alg(config, datasets, state.params, subkey, log_joints=self.log_joints, use_first_iter_cfg=False)
        state = state.replace(
            training_algs=tuple(a1.replace(config=a2.config) for a1, a2 in zip(state.training_algs, training_algs))
        )
        if all(isinstance(training_algs[i], MCMCAlgorithm) for i in range(len(training_algs))):
            assert all(isinstance(state.training_algs[i], MCMCAlgorithm) for i in range(len(state.training_algs)))
            state = state.replace(
                training_algs=tuple(a1.replace(_single_chains=a1._single_chains.replace(config=a2._single_chains.config)) for a1, a2 in zip(state.training_algs, training_algs))  # type: ignore
            )

        last_n_stable_states.append(state)
        _, stats = output
        # if config.recording_enabled:
        #     # whole trainstate is very heavy
        #     outputs.append([state, stats])
        # else:
        #     outputs.append([MiniTrainState(state.params), stats])

        if (
            config.likelihood_estimation_config.enabled
            and config.ebm_model_type == "joint_tilted"
        ):
            _, stats = output
            first_train_log_l = stats.loss["train_log_l"]
            best_test_log_l = stats.loss["test_log_l"]
        else:
            first_train_log_l = jnp.nan
            best_test_log_l = jnp.nan

        num_iter_no_test_increase = 0

        keys = random.split(key, num=config.max_iter - 1)

        iter_no = 1

        for key in keys:
            prev_state = state
            iter_no += 1
            if state.has_nan:
                print("encountered NaNs")
                print(state)
                break

            if state.opt_is_diverging:
                print(f"iter_no {iter_no}: opt is diverging")
                break

            if (
                config.restart_every is not None
                and (iter_no % config.restart_every) == 0
            ):
                key, key_init = random.split(key)
                init_state = self.initialize_state(datasets, config, key_init)
                state = state.replace(
                    training_algs=init_state.training_algs,
                    log_Z_alg=init_state.log_Z_algs,
                    opt_state=init_state.opt_state,
                )
                sampling_config_override = config.sampling_cfg_first_iter.replace(
                    num_steps=1000
                )
            else:
                sampling_config_override = None

            key, subkey = random.split(key)
            batched_datasets = self.get_batches(datasets, config.batch_size, subkey)

            key, key_train_step = random.split(key)

            t0 = time()
            state, output = jitted_train_step(
                state,
                batched_datasets,
                config.replace(update_all_particles=False),
                key_train_step,
                datasets,
            )
            # _alg = cast(TrainState, state).training_algs[0]
            # if isinstance(_alg, AdaptiveSMC):
            #     last_state = _alg._init_smc_state
            #     assert last_state is not None
            #     if (cast(TrainState, state).step % 100) == 0:
            #         print(f"step_size {jnp.mean(last_state.step_sizes):.3f}")
            # __import__('pdb').set_trace()
            _ = maybe_print_info(state, config, output[1])
            # print('train step time', time() - t0)

            if (
                config.likelihood_estimation_config.enabled
                and config.ebm_model_type == "joint_tilted"
            ):
                _, stats = output
                this_train_log_l = stats.loss["train_log_l"]
                this_test_log_l = stats.loss["test_log_l"]
                if (
                    state.step > config.n_iter_warmup
                    and this_train_log_l < first_train_log_l
                ):
                    print(
                        "likelihood is lower than the initial likelihood: the "
                        "optimization seems to be diverging"
                    )
                    state = state.replace(opt_is_diverging=True)

                if config.select_based_on_test_loss:
                    if this_test_log_l > best_test_log_l:
                        best_test_log_l = this_test_log_l
                        best_state = prev_state
                        if iter_no > config.n_iter_warmup:
                            num_iter_no_test_increase = 0

                    else:
                        if iter_no > config.n_iter_warmup:
                            num_iter_no_test_increase += 1

                            if num_iter_no_test_increase > config.patience:
                                best_state = best_state.replace(has_converged=True)
                                break

            if False:
                key, key_replay_buf = random.split(key)
                state = self._update_replay_buffer(
                    state, config, datasets, key_replay_buf
                )

            if state.step % max(config.max_iter // config.max_num_recordings, 1) == 0:
                state, stats = output
                if config.recording_enabled:
                    # whole trainstate is very heavy
                    outputs.append([state, stats])
                else:
                    outputs.append([MiniTrainState(state.params), stats])

            if (
                config.checkpoint_every is not None
                and (iter_no % config.checkpoint_every) == 0
            ):
                with open(f"checkpoint_{iter_no}.pkl", "wb") as f:
                    cloudpickle.dump(state, f)


            if state.step % 20 == 0:
                # print("appending")
                last_n_stable_states.append(state)

        trajectory = tree_map(lambda *args: jnp.stack(args), *outputs)

        if not config.select_based_on_test_loss:
            if state.opt_is_diverging:
                print("optimisation was diverging, using latest stable state")
                best_state = last_n_stable_states.popleft()
                print(best_state.step)
            else:
                best_state = state
            final_state = state
        else:
            final_state = best_state

        t = config.ebm_model_type
        if t in ("joint_unbiased", "likelihood", "joint_tilted", "joint_tilted_discrete"):
            ratio = None
            likelihood_factory = LikelihoodFactory(best_state.params, config.ebm)
            if t in ("likelihood", "joint_unbiased"):
                # if model type is "likelihood" but we're using LZNet, we're singly intractable
                if not (t == "likelihood" and config.estimate_log_normalizer):
                    likelihood_factory = likelihood_factory.replace(is_doubly_intractable=True)
        else:
            likelihood_factory = None
            assert len(self.log_joints) == 1
            assert isinstance(self.log_joints[0], (_EBMDiscreteJointDensity,))
            assert config.ebm_model_type == "ratio"
            config.ebm_model_type
            ratio = self.log_joints[0].ratio.replace(params=best_state.params)

        results = TrainerResults(
            init_state,
            final_state,
            best_state,
            trajectory[0],
            trajectory[1],
            # TODO: don't log the dataset in the results to limit memory usage and the size
            # of the results
            # datasets,
            None,
            config,
            likelihood_factory=likelihood_factory,
            ratio=ratio,
            time=time() - t0_init_training,
        )
        import datetime
        print("training ebm took time", str(datetime.timedelta(seconds=int(time() - t0_init_training))))
        return results
