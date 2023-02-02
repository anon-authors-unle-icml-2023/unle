from typing import (Tuple, Union, cast)

import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions

from sbi_ebm.data import SBIDataset, SBIParticles
from sbi_ebm.distributions import ThetaConditionalLogDensity
from sbi_ebm.likelihood_ebm import (Trainer, TrainingConfig)
from sbi_ebm.likelihood_trainer import LikelihoodTrainer
from sbi_ebm.pytypes import (PRNGKeyArray, PyTreeNode)
from sbi_ebm.samplers.inference_algorithms.base import InferenceAlgorithm
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm, MCMCAlgorithmFactory
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation

# jit = lambda x: x
from .likelihood_ebm import _EBMDiscreteJointDensity, _EBMMixedJointLogDensity, _EBMLikelihoodLogDensity, _EBMJointLogDensity, EBMLikelihoodConfig, LikelihoodEstimationConfig, TrainState, TrainingStats, energy, tree_any

from time import time
from typing import (Callable, Dict, Literal, NamedTuple,
                    Optional, Tuple, Type, TypeVar, Union, cast)

from numpyro import distributions as np_distributions
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
from sbi_ebm.distributions import (DoublyIntractableJointLogDensity, MixedJointLogDensity, ThetaConditionalLogDensity,
                                   maybe_wrap, maybe_wrap_joint)
from sbi_ebm.neural_networks import MLP, IceBeem
from sbi_ebm.pytypes import (Array, LogDensity_T, Numeric, PRNGKeyArray, PyTreeNode)
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm, MCMCAlgorithmFactory
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation
from sbi_ebm.samplers.kernels.discrete_gibbs import DiscreteLogDensity
from sbi_ebm.samplers.inference_algorithms.base import InferenceAlgorithm, InferenceAlgorithmFactory, InferenceAlgorithmInfo
from sbi_ebm.samplers.inference_algorithms.importance_sampling.smc import SMC, SMCFactory, SMCParticleApproximation
from sbi_ebm.train_utils import LikelihoodMonitor

# jit = lambda x: x


def maybe_reshape(x):
    import jax.numpy as jnp
    if len(x.shape) >=3:
        return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    elif len(x.shape) >= 2:
        return jnp.reshape(x, (x.shape[0] * x.shape[1],))
    else:
        raise ValueError("Can't reshape")



class TiltedLikelihoodTrainer(LikelihoodTrainer):
    def _init_training_alg(
        self,
        config: TrainingConfig,
        datasets: Tuple[SBIDataset, ...],
        params: PyTreeNode,
        key: PRNGKeyArray,
        log_joints: Tuple[Union[_EBMJointLogDensity, _EBMDiscreteJointDensity]],
        use_first_iter_cfg: bool = False,
    ) -> Tuple[InferenceAlgorithm, ...]:
        assert config.sampling_init_dist is not None  # type narrowing
        assert len(datasets) == 1
        assert len(log_joints) == 1
        dataset = datasets[0]
        log_joint = log_joints[0]

        if isinstance(config.sampling_init_dist, np_distributions.Distribution):
            # for likelihood-based training, we keep track of a particle xⁱ
            # per training point Θⁱ (sampled from p(x|Θⁱ; ψ)), but update
            # only `num_particles` per iteration.
            key, subkey = random.split(key)
            particles = config.sampling_init_dist.sample(
                # key=subkey, sample_shape=(dataset.train_samples.num_samples,)
                key=subkey, sample_shape=(dataset.train_samples.num_samples,)
            )
            assert particles.shape[1] == dataset.train_samples.dim_observations
        else:
            particles = dataset.train_samples.observations

        assert not isinstance(log_joint, _EBMDiscreteJointDensity)
        likelihoods = ThetaConditionalLogDensity(
            log_joint.log_likelihood.replace(params=params),
            dataset.train_samples.params
        )
        assert isinstance(config.sampling_cfg, MCMCAlgorithmFactory)

        if use_first_iter_cfg:
            factory = config.sampling_cfg_first_iter
        else:
            factory = config.sampling_cfg

        # _in_axes = tree_map(
        #     lambda x: _EBMLikelihoodLogDensity(None, None) if isinstance(x, _EBMLikelihoodLogDensity) else 0,  # type: ignore
        #     this_iter_algs, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        # )
        from sbi_ebm.samplers.inference_algorithms.mcmc.base import _MCMCChain
        _mcmc_axes = _MCMCChain(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None)  # pyright: ignore [reportGeneralTypeIssues]
        algs = vmap(
            type(factory).build_algorithm,
            in_axes=(None, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0)),  # type: ignore
            out_axes=MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None, _mcmc_axes),  # type: ignore
        )(config.sampling_cfg.replace(config=factory.config.replace(num_samples=1, num_chains=1)), likelihoods)

        assert isinstance(algs, MCMCAlgorithm)
        algs = vmap(
            type(algs).init_from_particles,
            in_axes=(MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None, _mcmc_axes), 0),  # type: ignore
            out_axes=MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), 0, _mcmc_axes.replace(_init_state=0))  # type: ignore
        )(
            algs, particles[:, None, :]
        )
        return (algs,)

    def compute_ebm_approx(self, alg: InferenceAlgorithm, log_joint: Union[_EBMDiscreteJointDensity, _EBMJointLogDensity], params: PyTreeNode, key: PRNGKeyArray, true_samples: SBIParticles) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        num_particles = min(self.num_particles, true_samples.num_samples)

        assert isinstance(alg.log_prob, ThetaConditionalLogDensity)
        assert isinstance(alg.log_prob.log_prob, _EBMLikelihoodLogDensity)
        alg = alg.set_log_prob(log_prob=alg.log_prob.replace(log_prob=alg.log_prob.log_prob.replace(params=params)))

        this_iter_algs = cast(
            type(alg),
            tree_map(
                lambda x: x if isinstance(x, _EBMLikelihoodLogDensity) else x[true_samples.indices[:num_particles]] ,
                alg, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity))
        )

        key, subkey = random.split(key)
        subkeys = random.split(subkey, num_particles)
        _in_axes = tree_map(
            lambda x: _EBMLikelihoodLogDensity(None, None) if isinstance(x, _EBMLikelihoodLogDensity) else 0,  # type: ignore
            this_iter_algs, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        )
        nta, results = vmap(
            type(this_iter_algs).run_and_update_init,
            # in_axes=(MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), 0, 0), 0))(
            in_axes=(_in_axes, 0))(
            this_iter_algs, subkeys
        )

        ebm_samples_xs = cast(
            ParticleApproximation, tree_map(lambda x: x[:, 0, ...], results.samples)
        )

        # use only samples returned by the mcmc algorithms and not the entire chain...
        ebm_samples_xs =ebm_samples_xs.replace(particles=jnp.concatenate([true_samples.params[:num_particles], ebm_samples_xs.particles, ], axis=1))

        ebm_samples_xs = cast(
            ParticleApproximation, tree_map(lambda x: x[:, 0, ...], results.samples)
        )
        ebm_samples_xs =ebm_samples_xs.replace(particles=jnp.concatenate([true_samples.params[:num_particles], ebm_samples_xs.particles, ], axis=1))


        # ...vs concatenate all iterations of MCMC chains
        # assert isinstance(results, MCMCResults)
        # ebm_traj = results.info.single_chain_results.chain.x
        # ebm_traj = tree_map(lambda x: x[:, 0, ...], ebm_traj)

        # x_and_thetas_traj = jnp.concatenate(
        #     [jnp.broadcast_to(true_samples.params[:num_particles, None, :], ebm_traj.shape[:-1] + (true_samples.params.shape[-1],)),
        #      ebm_traj], axis=-1
        # )
        # # ebm_traj = results.samples.replace(particles=x_and_thetas_traj, log_ws=ebm_traj.log_ws[:, -50:, 0, ...])
        # ebm_traj = results.samples.replace(particles=x_and_thetas_traj)
        # ebm_traj = tree_map(maybe_reshape, ebm_traj)
        # ebm_traj = ebm_traj.replace(log_ws=jnp.zeros((len(ebm_traj.particles),)))

        # # print(ebm_traj.particles.shape)
        # ebm_samples_xs  = ebm_traj


        # insert updated slice into original vmapped alg
        updated_alg = tree_map(
            lambda x, y: x if isinstance(x, _EBMLikelihoodLogDensity) else x.at[true_samples.indices[:num_particles]].set(y),
            alg,
            nta,
            is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        )
        return updated_alg, ebm_samples_xs

    def train_step(
        self,
        state: TrainState,
        datasets: Tuple[SBIDataset, ...],
        prior_dataset: SBIDataset,
        config: TrainingConfig,
        key: PRNGKeyArray,
        entire_datasets: Tuple[SBIDataset, ...],
        entire_prior_dataset: SBIDataset,
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
            tilted_likelihood_training_alg = state.tilted_likelihood_training_alg
            tilted_joint_training_alg = state.tilted_joint_training_alg
            assert tilted_likelihood_training_alg is not None
            assert tilted_joint_training_alg is not None
            new_tilted_joint_training_alg = tilted_joint_training_alg
            new_tilted_likelihood_training_alg = tilted_likelihood_training_alg

            key, subkey = random.split(key)
            new_tilted_joint_training_alg, tilted_results_joint = self.compute_tilted_joint_ebm_approx(
                tilted_joint_training_alg, cast(_EBMJointLogDensity, self.tilted_log_joint), state.params, subkey
            )

            key, subkey = random.split(key)
            new_tilted_likelihood_training_alg, tilted_results_likelihood = self.compute_tilted_likelihood_ebm_approx(
                tilted_likelihood_training_alg, cast(_EBMJointLogDensity, self.tilted_log_joint), state.params, subkey, prior_dataset.train_samples
            )

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
                    tilted_ebm_samples_likelihood=tilted_results_likelihood,
                    tilted_ebm_samples_joint=tilted_results_joint,
                    prior_samples=prior_dataset.train_samples,
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
            print('cold starting joint/likelihood training algs')
            key, key_particles = random.split(key)
            new_tilted_joint_training_alg = self._init_tilted_joint_training_alg(
                config, entire_prior_dataset, params, key_particles, self.tilted_log_joint
            )
            key, key_particles = random.split(key)
            new_tilted_likelihood_training_alg = self._init_tilted_likelihood_training_algs(
                config, entire_prior_dataset, params, key_particles, cast(Union[_EBMJointLogDensity, _EBMDiscreteJointDensity], self.tilted_log_joint)
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
            tilted_likelihood_training_alg=new_tilted_likelihood_training_alg,
            tilted_joint_training_alg=new_tilted_joint_training_alg,
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

    def compute_tilted_joint_ebm_approx(self, alg: InferenceAlgorithm, tilted_log_joint: Union[_EBMDiscreteJointDensity, _EBMJointLogDensity], params: PyTreeNode, key: PRNGKeyArray) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        alg = alg.set_log_prob(tilted_log_joint.set_params(params=params))
        # call the class method to prevent spurious recompilations.
        key, subkey = random.split(key)
        alg, results = type(alg).run_and_update_init(alg, subkey)
        return alg, results.samples

    def compute_tilted_likelihood_ebm_approx(self, alg: InferenceAlgorithm, tilted_log_joint: Union[_EBMDiscreteJointDensity, _EBMJointLogDensity], params: PyTreeNode, key: PRNGKeyArray, prior_samples: SBIParticles) -> Tuple[InferenceAlgorithm, ParticleApproximation]:
        num_particles = min(self.num_particles, prior_samples.num_samples)

        assert isinstance(alg.log_prob, ThetaConditionalLogDensity)
        assert isinstance(alg.log_prob.log_prob, _EBMLikelihoodLogDensity)
        alg = alg.set_log_prob(log_prob=alg.log_prob.replace(log_prob=alg.log_prob.log_prob.replace(params=params)))

        this_iter_algs = cast(
            type(alg),
            tree_map(
                lambda x: x if isinstance(x, _EBMLikelihoodLogDensity) else x[prior_samples.indices[:num_particles]] ,
                alg, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity))
        )

        key, subkey = random.split(key)
        subkeys = random.split(subkey, num_particles)
        _in_axes = tree_map(
            lambda x: _EBMLikelihoodLogDensity(None, None) if isinstance(x, _EBMLikelihoodLogDensity) else 0,  # type: ignore
            this_iter_algs, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        )
        nta, results = vmap(
            type(this_iter_algs).run_and_update_init,
            # in_axes=(MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), 0, 0), 0))(
            in_axes=(_in_axes, 0))(
            this_iter_algs, subkeys
        )

        ebm_samples_xs = cast(
            ParticleApproximation, tree_map(lambda x: x[:, 0, ...], results.samples)
        )

        # use only samples returned by the mcmc algorithms and not the entire chain...
        ebm_samples_xs =ebm_samples_xs.replace(particles=jnp.concatenate([prior_samples.params[:num_particles], ebm_samples_xs.particles, ], axis=1))

        ebm_samples_xs = cast(
            ParticleApproximation, tree_map(lambda x: x[:, 0, ...], results.samples)
        )
        ebm_samples_xs =ebm_samples_xs.replace(particles=jnp.concatenate([prior_samples.params[:num_particles], ebm_samples_xs.particles, ], axis=1))


        # ...vs concatenate all iterations of MCMC chains
        # assert isinstance(results, MCMCResults)
        # ebm_traj = results.info.single_chain_results.chain.x
        # ebm_traj = tree_map(lambda x: x[:, 0, ...], ebm_traj)

        # x_and_thetas_traj = jnp.concatenate(
        #     [jnp.broadcast_to(true_samples.params[:num_particles, None, :], ebm_traj.shape[:-1] + (true_samples.params.shape[-1],)),
        #      ebm_traj], axis=-1
        # )
        # # ebm_traj = results.samples.replace(particles=x_and_thetas_traj, log_ws=ebm_traj.log_ws[:, -50:, 0, ...])
        # ebm_traj = results.samples.replace(particles=x_and_thetas_traj)
        # ebm_traj = tree_map(maybe_reshape, ebm_traj)
        # ebm_traj = ebm_traj.replace(log_ws=jnp.zeros((len(ebm_traj.particles),)))

        # # print(ebm_traj.particles.shape)
        # ebm_samples_xs  = ebm_traj


        # insert updated slice into original vmapped alg
        updated_alg = tree_map(
            lambda x, y: x if isinstance(x, _EBMLikelihoodLogDensity) else x.at[prior_samples.indices[:num_particles]].set(y),
            alg,
            nta,
            is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        )
        return updated_alg, ebm_samples_xs


    def initialize_state(
        self,
        datasets: Tuple[SBIDataset, ...],
        prior_dataset: SBIDataset,
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

        log_joints, tilted_log_joint = self._make_log_joint(params, config, datasets, prior_dataset, calibration_net)

        # 3.a PARTICLE APPROXIMATION (gradient)
        key, key_init_particles = random.split(key)
        training_algs = self._init_training_alg(config, datasets, params, key_init_particles, cast(Tuple[_EBMJointLogDensity], log_joints), use_first_iter_cfg)

        # 3.a JOINT TILTED PARTICLE APPROXIMATION (gradient)
        key, key_init_particles = random.split(key)
        joint_tilted_training_algs = self._init_tilted_joint_training_alg(config, prior_dataset, params, key_init_particles, tilted_log_joint, use_first_iter_cfg)

        # 3.a TILTED PARTICLE APPROXIMATIONs (gradient)
        key, key_init_particles = random.split(key)

        tilted_likelihood_training_algs = self._init_tilted_likelihood_training_algs(config, prior_dataset, params, key_init_particles, cast(_EBMJointLogDensity,  tilted_log_joint), use_first_iter_cfg)

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
            tilted_joint_training_alg=joint_tilted_training_algs,
            tilted_likelihood_training_alg=tilted_likelihood_training_algs,
        )
        return state

    def _init_tilted_likelihood_training_algs(
        self,
        config: TrainingConfig,
        prior_dataset: SBIDataset,
        params: PyTreeNode,
        key: PRNGKeyArray,
        tilted_log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity],
        use_first_iter_cfg: bool = False,
    ) -> InferenceAlgorithm:
        assert config.sampling_init_dist is not None  # type narrowing

        if isinstance(config.sampling_init_dist, np_distributions.Distribution):
            # for likelihood-based training, we keep track of a particle xⁱ
            # per training point Θⁱ (sampled from p(x|Θⁱ; ψ)), but update
            # only `num_particles` per iteration.
            key, subkey = random.split(key)
            particles = config.sampling_init_dist.sample(
                # key=subkey, sample_shape=(dataset.train_samples.num_samples,)
                key=subkey, sample_shape=(prior_dataset.train_samples.num_samples,)
            )
            assert particles.shape[1] == prior_dataset.train_samples.dim_observations
        else:
            particles = prior_dataset.train_samples.observations

        assert not isinstance(tilted_log_joint, _EBMDiscreteJointDensity)
        likelihoods = ThetaConditionalLogDensity(
            tilted_log_joint.log_likelihood.replace(params=params),
            prior_dataset.train_samples.params,
        )
        assert isinstance(config.sampling_cfg, MCMCAlgorithmFactory)

        if use_first_iter_cfg:
            factory = config.sampling_cfg_first_iter
        else:
            factory = config.sampling_cfg

        # _in_axes = tree_map(
        #     lambda x: _EBMLikelihoodLogDensity(None, None) if isinstance(x, _EBMLikelihoodLogDensity) else 0,  # type: ignore
        #     this_iter_algs, is_leaf=lambda x: isinstance(x, _EBMLikelihoodLogDensity)
        # )
        from sbi_ebm.samplers.inference_algorithms.mcmc.base import _MCMCChain
        _mcmc_axes = _MCMCChain(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None)  # pyright: ignore [reportGeneralTypeIssues]
        algs = vmap(
            type(factory).build_algorithm,
            in_axes=(None, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0)),  # type: ignore
            out_axes=MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None, _mcmc_axes),  # type: ignore
        )(config.sampling_cfg.replace(config=factory.config.replace(num_samples=1, num_chains=1)), likelihoods)

        assert isinstance(algs, MCMCAlgorithm)
        algs = vmap(
            type(algs).init_from_particles,
            in_axes=(MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), None, _mcmc_axes), 0),  # type: ignore
            out_axes=MCMCAlgorithm(0, ThetaConditionalLogDensity(_EBMLikelihoodLogDensity(None, None), 0), 0, _mcmc_axes.replace(_init_state=0))  # type: ignore
        )(
            algs, particles[:, None, :]
        )
        return algs

    def _resolve_tilted_joint_proposal_distribution(self, config: TrainingConfig, datasets: Tuple[SBIDataset]) -> Tuple[np_distributions.Distribution]:
        if config.ebm_model_type == "ratio":
            log_density_uniform = np_distributions.DiscreteUniform(low=jnp.zeros((2,)), high=len(datasets[0].train_samples.observations) * jnp.ones((2,))).to_event()
            return tuple(log_density_uniform for _ in range(len(datasets)))
        else:
            assert isinstance(config.sampling_init_dist, np_distributions.Distribution)
            return tuple(config.sampling_init_dist for _ in range(len(datasets)))



    def _init_tilted_joint_training_alg(
        self,
        config: TrainingConfig,
        prior_dataset: SBIDataset,
        params: PyTreeNode,
        key: PRNGKeyArray,
        tilted_log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity, _EBMMixedJointLogDensity],
        use_first_iter_cfg: bool = False,
        # algs: Optional[Tuple[InferenceAlgorithm]] = None,
    ) -> InferenceAlgorithm:
        assert config.num_particles is not None  # type narrowing
        assert config.sampling_init_dist is not None  # type narrowing
        assert config.sampling_config_tilted_first_iter is not None
        assert config.sampling_config_tilted is not None
        assert config.sampling_config_tilted_init_dist is not None
        if use_first_iter_cfg:
            alg = config.sampling_config_tilted_first_iter.build_algorithm(log_prob=tilted_log_joint.set_params(params))
        else:
            alg = config.sampling_config_tilted.build_algorithm(log_prob=tilted_log_joint.set_params(params))

        if isinstance(config.sampling_config_tilted_init_dist, np_distributions.Distribution):
            dists = self._resolve_proposal_distribution(config, (prior_dataset,))
            dist = dists[0]
            key, subkey = random.split(key)
            alg = alg.init(fold_in(subkey, 0), dist)
        else:
            particles = self._resolve_proposal_particles(config, (prior_dataset,), key)
            assert isinstance(alg, MCMCAlgorithm)
            alg = alg.init_from_particles(particles[0])
        return alg

    @staticmethod
    def tilting_penalty(params, tilted_ebm_samples_likelihood: ParticleApproximation, tilted_ebm_samples_joint: ParticleApproximation, prior_samples: SBIParticles, ebm_config: EBMLikelihoodConfig, noise_injection_val: float, key: PRNGKeyArray):
        dim_z = prior_samples.dim_params
        def energy_fn(z, x):
            # the total "energy" of the joint is (minus) the joint log-probability.
            # However, the prior and base measure do not depend on the neural
            # network: the only remaining term that actually has a gradient is the
            # energy network itself.
            return energy(ebm_config.energy_network_type, ebm_config.width, ebm_config.depth).apply(
                params, (z, x)
            )

        noise: Array = noise_injection_val * random.normal(
            key, prior_samples.xs.shape
        )

        energy_tilted_samples_likelihood = jnp.average(
            vmap(energy_fn)(
                tilted_ebm_samples_likelihood.xs[:, :dim_z],  # + noise[:, :dim_z],
                tilted_ebm_samples_likelihood.xs[:, dim_z:]
            ),
            weights=prior_samples.normalized_ws,
        )

        energy_tilted_joint_samples = jnp.average(
            vmap(energy_fn)(tilted_ebm_samples_joint.xs[:, :dim_z], tilted_ebm_samples_joint.xs[:, dim_z:]),
            weights=tilted_ebm_samples_joint.normalized_ws,
        )

        return (
            energy_tilted_samples_likelihood
            - energy_tilted_joint_samples
            # + 0.01 * (energy_ebm_samples ** 2 + energy_ebm_samples ** 2)
            # + logsumexp(energy_ebm_samples) + logsumexp(-energy_ebm_samples)
            # + 100 * (jax.nn.logsumexp(energy_ebm_samples) + jax.nn.logsumexp(-energy_ebm_samples))
        )



    def estimate_tilting_penalty_gradient(
        self,
        params: PyTreeNode,
        tilted_ebm_samples_likelihood: ParticleApproximation,
        tilted_ebm_samples_joint: ParticleApproximation,
        prior_samples: SBIParticles,
        ebm_config: EBMLikelihoodConfig,
        noise_injection_val: float,
        key: PRNGKeyArray,
        log_joint: Union[_EBMJointLogDensity, _EBMDiscreteJointDensity],
    ) -> PyTreeNode:
        objective_gradient = grad(self.tilting_penalty)(params, tilted_ebm_samples_likelihood, tilted_ebm_samples_joint, prior_samples, ebm_config, noise_injection_val, key)
        return objective_gradient

    def estimate_value_and_grad(
        self,
        params: PyTreeNode,
        ebm_config: EBMLikelihoodConfig,
        noise_injection_val: float,
        proposal_log_prob: LogDensity_T,
        ebm_samples: ParticleApproximation,
        ebm_samples_log_Z: Optional[SMCParticleApproximation],
        tilted_ebm_samples_likelihood: ParticleApproximation,
        tilted_ebm_samples_joint: ParticleApproximation,
        prior_samples: SBIParticles,
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

        key, subkey = random.split(key)
        grads_tilted = self.estimate_tilting_penalty_gradient(
            params,
            tilted_ebm_samples_likelihood,
            tilted_ebm_samples_joint,
            prior_samples,
            ebm_config,
            noise_injection_val,
            subkey,
            log_joint,
        )

        print('adding grads!')
        grads = jax.tree_map(lambda x, y: x + y, grads, grads_tilted)

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

