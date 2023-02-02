from sbi_ebm.pytypes import LogDensity_T, PRNGKeyArray
from sbi_ebm.samplers.inference_algorithms.mcmc.base import MCMCConfig, MCMCResults, sample
from sbi_ebm.samplers.particle_aproximation import ParticleApproximation

from ...kernels.mala import MALAConfig, MALAInfo, MALAKernel, MALAState
from ...kernels.ula import ULAConfig, ULAInfo, ULAKernel, ULAState


def vmapped_ula(
    log_prob: LogDensity_T,
    x0s: ParticleApproximation,
    config: MCMCConfig[ULAConfig, ULAState, ULAInfo],
    key: PRNGKeyArray,
) -> MCMCResults[ULAState, ULAInfo]:
    return sample(log_prob, x0s, config, key)


def vmapped_mala(
    log_prob: LogDensity_T,
    x0s: ParticleApproximation,
    config: MCMCConfig[MALAConfig, MALAState, MALAInfo],
    key: PRNGKeyArray,
) -> MCMCResults[MALAState, MALAInfo]:
    return sample(log_prob, x0s, config, key)
