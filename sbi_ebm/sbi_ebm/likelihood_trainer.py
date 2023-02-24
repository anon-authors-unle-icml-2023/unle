from typing import (Tuple, Union, cast)

import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions

from sbi_ebm.data import SBIDataset, SBIParticles
from sbi_ebm.distributions import ThetaConditionalLogDensity
from sbi_ebm.likelihood_ebm import (Trainer, TrainingConfig)
from sbi_ebm.pytypes import (PRNGKeyArray, PyTreeNode)
from sbi_ebm.samplers.inference_algorithms.base import InferenceAlgorithm
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCAlgorithm, MCMCAlgorithmFactory
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation

# jit = lambda x: x
from .likelihood_ebm import _EBMDiscreteJointDensity, _EBMLikelihoodLogDensity, _EBMJointLogDensity


def maybe_reshape(x):
    import jax.numpy as jnp
    if len(x.shape) >=3:
        return jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    elif len(x.shape) >= 2:
        return jnp.reshape(x, (x.shape[0] * x.shape[1],))
    else:
        raise ValueError("Can't reshape")



class LikelihoodTrainer(Trainer):
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
