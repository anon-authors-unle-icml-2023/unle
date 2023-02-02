from typing import List

import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.linear import Dense
from flax.linen.module import Module
from flax.linen.normalization import BatchNorm
from jax.nn import softplus
from numpyro import distributions as np_distributions
from numpyro.distributions import transforms as np_transforms
from pyro.distributions import transforms as pyro_transforms
from pyro.distributions.transforms import identity_transform
from torch.nn import Module as torch_Module
from torch.nn import Softplus
from sbi_ebm.distributions import maybe_wrap_log_l

from sbi_ebm.pytypes import Array, PyTreeNode

# This module is not capable of converting the scalers, only the neural
# networks, so the likelihood should always be used with "scaled" inputs.


def _build_sequential_ops(torch_nn):
    ops = []
    ops.append(torch_nn._modules["fc_in"])
    ops.append(torch_nn._modules["nonlinearity_in"])

    hidden_ops = list(torch_nn._modules["fc_hidden"]._modules.values())
    hidden_nonls = list(torch_nn._modules["nonlinearities_hidden"]._modules.values())

    for layer, non_linearity in zip(hidden_ops, hidden_nonls):
        ops.append(layer)
        ops.append(non_linearity)
    ops.append(torch_nn._modules["fc_out"])
    return ops


def _build_sequential_ops_batchnorm(torch_nn):
    ops = []
    ops.append(torch_nn._modules["fc_in"])

    assert "nonlinearity_in" not in torch_nn._modules
    ops.append(Softplus())

    hidden_ops = list(torch_nn._modules["fc_hidden"]._modules.values())

    assert "nonlinearites_hidden" not in torch_nn._modules
    hidden_ops = list(torch_nn._modules["fc_hidden"]._modules.values())
    for layer in hidden_ops:
        ops.append(layer)
        ops.append(Softplus())
    ops.append(torch_nn._modules["fc_out"])

    if "bn_out" in torch_nn._modules:
        ops.append(torch_nn._modules["bn_out"])
    return ops


class SMNLEModule(Module):
    module: torch_Module
    converter_map = {"Linear": Dense, "Softplus": softplus, "BatchNorm1d": BatchNorm}
    batch_norm: bool

    def setup(self):
        if not self.batch_norm:
            self._jax_sequential_ops = self._to_jax(_build_sequential_ops(self.module))
        else:
            self._jax_sequential_ops = self._to_jax(
                _build_sequential_ops_batchnorm(self.module)
            )

    @property
    def torch_sequential_ops(self):
        if not self.batch_norm:
            return _build_sequential_ops(self.module)
        else:
            return _build_sequential_ops_batchnorm(self.module)

    def _to_jax(self, ops):

        jax_ops = []
        for op in ops:
            op_name = type(op).__name__
            assert op_name in self.converter_map, op_name
            if op_name == "Linear":
                jax_op = Dense(op.out_features)
            elif op_name == "Softplus":
                assert op.beta == 1
                jax_op = softplus
            elif op_name == "BatchNorm1d":
                jax_op = BatchNorm(use_running_average=True)
            else:
                raise ValueError(op_name)

            jax_ops.append(jax_op)
        return jax_ops

    def __call__(self, x):
        for op in self._jax_sequential_ops:
            x = op(x)
        return x


def convert_to_jax_state(jax_orig_params, module_list):
    new_params = {}

    if "batch_stats" in jax_orig_params:
        assert type(module_list[-1]).__name__ == "BatchNorm1d"
        assert len(jax_orig_params["batch_stats"].keys()) == 1
        batch_stats_param_name = list(jax_orig_params["batch_stats"].keys())[0]

        batch_stats = {}
        batch_stats[batch_stats_param_name] = {
            "mean": jnp.array(
                module_list[-1]._buffers["running_mean"].detach().numpy()
            ),
            "var": jnp.array(module_list[-1]._buffers["running_var"].detach().numpy()),
        }

        new_params["batch_stats"] = batch_stats

    neural_net_params = {}
    for d1, d2 in zip(
        jax_orig_params["params"].items(),
        [o for o in module_list if type(o).__name__ != "Softplus"],
    ):

        d1_name, d1_val = d1

        if "kernel" in d1_val:
            assert "bias" in d1_val

            torch_params = list(d2.parameters())
            assert len(torch_params) == 2
            neural_net_params[d1_name] = FrozenDict(
                {
                    "kernel": jnp.array(torch_params[0].T.detach().numpy()),
                    "bias": jnp.array(torch_params[1].detach().numpy()),
                }
            )

        elif "scale" in d1_val:
            assert "bias" in d1_val
            # additional bias and scale for the batch norm layer. Laxy
            # Passthrough as it composed of respectively 0s and 1s at init.
            neural_net_params[d1_name] = d1_val
        else:
            raise ValueError(d1_val)

    new_params["params"] = neural_net_params
    return FrozenDict(new_params)


class SMExpFamJax(Module):
    torch_net_data: torch_Module
    torch_net_theta: torch_Module

    def setup(self):
        self.net_data = SMNLEModule(self.torch_net_data, batch_norm=False)
        self.net_theta = SMNLEModule(self.torch_net_theta, batch_norm=True)

    def __call__(self, theta_and_x):
        theta, x = theta_and_x
        suff_stat = self.net_data(x)
        nat_param = self.net_theta(theta)

        nat_param_and_bias = jnp.concatenate([nat_param, jnp.array([1])])
        return jnp.dot(suff_stat, nat_param_and_bias)


class JaxExpFamLikelihood(struct.PyTreeNode):
    likelihood_module: SMExpFamJax = struct.field(pytree_node=False)
    params: PyTreeNode = struct.field(pytree_node=True)
    # theta_transform transforms theta to the z-scored space
    theta_transform: np_transforms.Transform = struct.field(
        pytree_node=False, default=np_transforms.IdentityTransform()
    )
    # ditto
    data_transform: np_transforms.Transform = struct.field(
        pytree_node=False, default=np_transforms.IdentityTransform()
    )

    def __call__(self, theta: Array, x: Array):
        return self.likelihood_module.apply(self.params, (theta, x))


class _JaxExpFamLikelihoodDist(np_distributions.Distribution):
    # https://github.com/pyro-ppl/numpyro/issues/1317
    # Distributions instances cannot be vmapped, and thus are not used during the
    # training loop. Instances of this class are return by sbi_ebm at the end of
    # training for ease of potential integration downstream numpyro applications.
    arg_contraints = {"likelihood_log_prob": None, "param": None}

    def __init__(self, likelihood_log_prob: JaxExpFamLikelihood, param: Array):
        self.likelihood_log_prob = likelihood_log_prob
        self.param = param

        import torch

        _fc_in = self.likelihood_log_prob.likelihood_module.torch_net_data.fc_in
        assert not isinstance(_fc_in, torch.Tensor)
        event_shape = (_fc_in.in_features,)
        super(_JaxExpFamLikelihoodDist, self).__init__(
            batch_shape=(), event_shape=event_shape
        )

    def log_prob(self, x):
        return self.likelihood_log_prob(self.param, x)


def make_jax_likelihood(
    net_data: torch_Module,
    net_theta: torch_Module,
    data_transforms: pyro_transforms.Transform = identity_transform,
    theta_transforms: pyro_transforms.Transform = identity_transform,
):
    nd = net_data
    nt = net_theta

    smd = SMNLEModule(nd, batch_norm=False)
    smt = SMNLEModule(nt, batch_norm=True)

    import jax.numpy as jnp
    from jax import random

    params_nd = smd.init(random.PRNGKey(0), jnp.zeros((2,)))
    params_nt = smt.init(random.PRNGKey(0), jnp.zeros((2,)))

    jax_params_nd = convert_to_jax_state(params_nd, smd.torch_sequential_ops)
    jax_params_nt = convert_to_jax_state(params_nt, smt.torch_sequential_ops)

    sme = SMExpFamJax(nd, nt)
    params_sme = sme.init(random.PRNGKey(0), (jnp.zeros((2,)), jnp.zeros((2,))))
    new_params_sme = {
        "params": {
            "net_data": jax_params_nd["params"],
            "net_theta": jax_params_nt["params"],
        },
    }
    if "batch_stats" in jax_params_nt:
        new_params_sme["batch_stats"] = {"net_theta": jax_params_nt["batch_stats"]}

    new_params_sme = FrozenDict(new_params_sme)

    liklelihood_zscored_space = JaxExpFamLikelihood(sme, new_params_sme)
    from sbi_ebm.sbibm.pyro_to_numpyro import convert_transform

    jax_theta_transform = convert_transform(theta_transforms, "numpyro")
    jax_data_transform = convert_transform(data_transforms, "numpyro")

    def log_likelihood(theta, x):
        return np_distributions.TransformedDistribution(
            _JaxExpFamLikelihoodDist(
                liklelihood_zscored_space, jax_theta_transform(theta)
            ),
            jax_data_transform.inv,
        ).log_prob(x)

    return maybe_wrap_log_l(log_likelihood)
