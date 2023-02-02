import abc
from typing import List, Literal, NamedTuple, Optional, Tuple, Type, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import random, vmap
from jax.tree_util import tree_map
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from numpyro.distributions.transforms import AffineTransform
from typing_extensions import Self

from sbi_ebm.pytypes import Array, PRNGKeyArray
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation


class SBIParticles(ParticleApproximation):
    # particles approximation with the following characteristics:
    # - each particle is composed of two random variables: parameter and an observed
    #   variable
    # - the prior approximated by the particles is known analytically
    prior: np_distributions.Distribution = struct.field(pytree_node=False)
    _dim_params: int = struct.field(pytree_node=False)
    indices: Array = struct.field(pytree_node=True, default=None)

    @property
    def params(self):
        return self.xs[:, : self._dim_params]

    @property
    def observations(self):
        return self.xs[:, self._dim_params :]

    @property
    def dim_params(self) -> int:
        return self.params.shape[1]

    @property
    def dim_observations(self) -> int:
        return self.observations.shape[1]

    @classmethod
    def create(
        cls: Type[Self],
        params: Array,
        observations: Array,
        prior: np_distributions.Distribution,
        log_ws: Optional[Array] = None,
    ) -> Self:
        assert len(params.shape) == len(observations.shape) == 2
        assert params.shape[0] == observations.shape[0]
        num_samples, dim_params = params.shape

        if log_ws is None:
            log_ws = jnp.zeros((num_samples,))

        xs = jnp.concatenate([params, observations], axis=1)
        indices = jnp.arange(num_samples)
        return cls(xs, log_ws, prior, dim_params, indices)


class SBIDataset(NamedTuple):
    train_samples: SBIParticles
    test_samples: SBIParticles

    @classmethod
    def create(
        cls: Type[Self],
        params: Array,
        observations: Array,
        prior: np_distributions.Distribution,
        frac_test_samples: float,
        key: Optional[PRNGKeyArray] = None,
    ) -> Self:
        if key is not None:
            key, subkey = random.split(key)
            params = random.permutation(subkey, params, axis=0)
            observations = random.permutation(subkey, observations, axis=0)

        samples = SBIParticles.create(
            params=params, observations=observations, prior=prior, log_ws=None
        )
        num_test_samples = int(frac_test_samples * samples.num_samples)

        train_samples: SBIParticles = tree_map(lambda a: a[num_test_samples:], samples)
        test_samples: SBIParticles = tree_map(lambda a: a[:num_test_samples], samples)
        return cls(train_samples, test_samples)

    @property
    def dim_params(self):
        return self.train_samples.dim_params

    @property
    def dim_observations(self):
        return self.train_samples.dim_observations

    @property
    def prior(self) -> np_distributions.Distribution:
        return self.train_samples.prior

    @property
    def observations(self) -> Array:
        return cast(Array, jnp.concatenate([self.train_samples.observations, self.test_samples.observations], axis=0))

    @property
    def params(self) -> Array:
        return cast(Array, jnp.concatenate([self.train_samples.params, self.test_samples.params], axis=0))



class ABCPyTreeNodeMeta(abc.ABCMeta, type(struct.PyTreeNode)):
    pass


class DataTransform(struct.PyTreeNode, metaclass=ABCPyTreeNodeMeta):
    @abc.abstractmethod
    def get_transform(
        self, who: Literal["params", "observations"]
    ) -> np_transforms.Transform:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def create_and_fit(cls: Type[Self], data: SBIParticles) -> Self:
        raise NotImplementedError

    def transform(self, samples: SBIParticles) -> SBIParticles:
        param_transform = self.get_transform("params")
        transformed_prior = np_distributions.TransformedDistribution(
            samples.prior, param_transform
        )
        return SBIParticles.create(
            params=param_transform(samples.params),
            observations=self.get_transform("observations")(samples.observations),
            prior=transformed_prior,
            log_ws=samples.log_ws,
        )

    # def transform(self, dataset: SBIDataset) -> SBIDataset:
    #     return SBIDataset(train_samples=self.transform(dataset.train_samples), test_samples=self.transform(dataset.test_samples))


class Normalizer(DataTransform):
    params_mean: Array
    params_std: Array
    observations_mean: Array
    observations_std: Array

    @classmethod
    def create_and_fit(cls, data: SBIParticles):
        params_mean = jnp.average(data.params, axis=0, weights=data.normalized_ws)
        observations_mean = jnp.average(
            data.observations, axis=0, weights=data.normalized_ws
        )
        params_std = jnp.sqrt(
            jnp.average(
                (data.params - params_mean) ** 2, axis=0, weights=data.normalized_ws
            )
        )
        observations_std = (
            jnp.sqrt(
                jnp.average(
                    (data.observations - observations_mean) ** 2,
                    axis=0,
                    weights=data.normalized_ws,
                )
            )
            + 1e-8
        )
        return cls(params_mean, params_std, observations_mean, observations_std)

    def get_transform(self, who: Literal["params", "observations"]):

        if who == "params":
            mean, std = self.params_mean, self.params_std
        elif who == "observations":
            mean, std = self.observations_mean, self.observations_std
        else:
            raise ValueError

        return AffineTransform(loc=mean, scale=std).inv


class Unconstrainer(DataTransform):
    inv_transform: np_transforms.Transform

    @classmethod
    def create_and_fit(cls, data: SBIParticles):
        inv_transform = np_transforms.biject_to(data.prior.support)
        return cls(inv_transform)

    def get_transform(self, who: Literal["params", "observations"]):
        if who == "params":
            return self.inv_transform.inv
        elif who == "observations":
            return np_transforms.IdentityTransform()
        else:
            raise ValueError


class ZScorer(DataTransform):
    data_transforms: Tuple[DataTransform, DataTransform]

    @classmethod
    def create_and_fit(
        cls: Type[Self],
        data: SBIParticles,
        normalize: bool,
        biject_to_unconstrained_space: bool,
    ):
        data_transforms: List[DataTransform] = []
        if biject_to_unconstrained_space:
            unconstrainer = Unconstrainer.create_and_fit(data)
            data = unconstrainer.transform(data)
            data_transforms.append(unconstrainer)

        if normalize:
            normalizer = Normalizer.create_and_fit(data)
            data_transforms.append(normalizer)

        data_transforms_tuple = tuple(data_transforms)

        return cls(data_transforms=data_transforms_tuple)

    def get_transform(self, who: Literal["params", "observations"]):
        transforms = [d.get_transform(who) for d in self.data_transforms]
        return np_transforms.ComposeTransform(transforms)
