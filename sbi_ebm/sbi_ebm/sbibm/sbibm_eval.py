import functools
from typing import Any, Callable, List, NamedTuple, Optional, TypeVar

from sbibm.algorithms import snle as sbibm_snle
from sbibm.algorithms import snpe as sbibm_snpe
from sbibm.algorithms import snre as sbibm_snre
from typing_extensions import ParamSpec

from sbi_ebm.sbibm.sbi_ebm import MetricResults
from sbi_ebm.sbibm.tasks import get_task

from .sbi_ebm import _evaluate_posterior

P = ParamSpec("P")
R = TypeVar("R")

class SBIBMSingleRoundResults(NamedTuple):
    posterior: Any
    posterior_samples: Any
    dataset: Any


class SBIBMTrainResults(NamedTuple):
    posterior: Any
    posterior_samples: Any
    config: dict
    single_round_results: List[SBIBMSingleRoundResults]


class SBIBMTrainEvalResults(NamedTuple):
    train_results: SBIBMTrainResults
    eval_results: Optional[MetricResults] = None


def wrap_sbibm_method(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapping_func(**kwargs):
        orig_kwargs = kwargs

        kwargs = orig_kwargs.copy()
        task_name = kwargs.pop("task")
        task = get_task(task_name)

        posterior, samples, _, single_round_results = func(task=task, **kwargs)

        single_round_results = [SBIBMSingleRoundResults(srr['posterior'], srr['posterior_samples'], srr['data']) for srr in single_round_results]

        eval_res = _evaluate_posterior(samples, task, kwargs["num_observation"])
        train_res = SBIBMTrainResults(posterior, samples, orig_kwargs, single_round_results)
        return SBIBMTrainEvalResults(train_results=train_res, eval_results=eval_res)

    return wrapping_func


def snre(**kwargs) -> SBIBMTrainEvalResults:
    return wrap_sbibm_method(sbibm_snre)(**kwargs)


def snpe(**kwargs) -> SBIBMTrainEvalResults:
    return wrap_sbibm_method(sbibm_snpe)(**kwargs)


def snle(**kwargs) -> SBIBMTrainEvalResults:
    return wrap_sbibm_method(sbibm_snle)(**kwargs)
