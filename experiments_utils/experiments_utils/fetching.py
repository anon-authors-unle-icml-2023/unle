import pickle
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
from joblib import Parallel, delayed

from experiments_utils.submission import SLURMJobMetadata
from .utils import _resolve_slurm_file_name


def fetch_single_result(p: Path, func: Callable):
    if p.name in ("results.pkl", "results-bak.pkl"):
        raise ValueError
    else:
        with open(p, "rb") as f:
            ret = pickle.load(f)

    from .submission import _get_defaults

    defs = _get_defaults(func).copy()
    defs.update(ret[0])
    return defs, p.name


def refetch_results_db(results_dir, func, **kwargs):
    kws_and_filenames = Parallel(n_jobs=30, backend="threading", verbose=5)(
        delayed(fetch_single_result)(p, func)
        for p in list(results_dir.iterdir())
        if p.name not in ("results.pkl", "results-bak.pkl")
    )
    kws_and_filenames = [k for k in kws_and_filenames if all(k[0].get(_qk, None) == _vk for _qk, _vk in kwargs.items())]
    kws_and_filenames = tuple(zip(*kws_and_filenames))

    index = pd.MultiIndex.from_tuples(
        tuple(kws.values() for kws in kws_and_filenames[0]),
        names=kws_and_filenames[0][0].keys(),
    )
    return pd.Series(kws_and_filenames[1], index=index)


class Result:
    def __init__(
        self,
        result_val,
        metadata_kwargs: Dict,
        all_kwargs: Dict,
        slurm_metadata: SLURMJobMetadata,
    ):
        self.result = result_val
        self.metadata_kwargs = metadata_kwargs
        self.all_kwargs = all_kwargs
        self.slurm_metadata = slurm_metadata
        self.ran_in_slurm = self.slurm_metadata.sbatch_script_filepath is not None

        self._slurm_out_file_name = None
        self._slurm_out_file_contents = None
        self._slurm_err_file_name = None
        self._slurm_err_file_contents = None

    @property
    def slurm_out_file_name(self):
        if not self.ran_in_slurm:
            return None
        if self._slurm_out_file_name is None:
            assert self.slurm_metadata.slurm_jobid is not None
            self._slurm_out_file_name = _resolve_slurm_file_name(
                str(self.slurm_metadata.sbatch_script_filepath), self.slurm_metadata.slurm_jobid, "output"
            )
        return self._slurm_out_file_name

    @property
    def slurm_err_file_name(self):
        if not self.ran_in_slurm:
            return None

        if self._slurm_err_file_name is None:
            assert self.slurm_metadata.slurm_jobid is not None
            self._slurm_err_file_name = _resolve_slurm_file_name(
                str(self.slurm_metadata.sbatch_script_filepath), self.slurm_metadata.slurm_jobid, "error"
            )
        return self._slurm_err_file_name

    @property
    def slurm_err_file_contents(self):
        slurm_err_file_name = self.slurm_err_file_name
        if slurm_err_file_name is None:
            return

        if self._slurm_err_file_contents is None:
            with open(slurm_err_file_name, 'r') as f:
                self._slurm_err_file_contents = f.readlines()
        return ''.join(self._slurm_err_file_contents)

    @property
    def slurm_out_file_contents(self):
        slurm_out_file_name = self.slurm_out_file_name
        if slurm_out_file_name is None:
            return

        if self._slurm_out_file_contents is None:
            with open(slurm_out_file_name, 'r') as f:
                self._slurm_out_file_contents = f.readlines()
        return ''.join(self._slurm_out_file_contents)



    def __repr__(self) -> str:
        dict_vals_fmt = ", ".join(f"{k}={v}" for k, v in self.metadata_kwargs.items())
        return f"Result({dict_vals_fmt})"


class ResultsManager:
    def __init__(self, folder_name):
        from experiments_utils.config import get_config
        self.abs_folder_name = get_config().results_dir / folder_name
        self._results_df = None
        self._minimal_results_df = None

    @property
    def results_df(self) -> pd.Series:
        if self._results_df is None:
            self._results_df = self._get_index_df(minimal=False)
        return self._results_df

    @property
    def minimal_results_df(self) -> pd.Series:
        if self._minimal_results_df is None:
            self._minimal_results_df = self._minify_df(self.results_df)
        return self._minimal_results_df

    def _get_index_df(self, minimal=False) -> pd.Series:
        with open(self.abs_folder_name / "results.pkl", "rb") as f:
            index = pickle.load(f)
        index = index.sort_index()

        if minimal:
            return self._minify_df(index)
        return index

    def _minify_df(self, df: pd.Series) -> pd.Series:
        if len(df) == 1:
            # nothing to minify
            return df

        # hide the index levels (in a multi-index) for which the size of the set of
        # unique values is 1.
        index_vals = df.reset_index().iloc[:, : df.index.nlevels]
        n_unique = index_vals.apply(lambda x: len(x.unique()), axis=0)

        levels_with_unique_vals = [n for n in df.index.names if n_unique[n] == 1]
        return df.reset_index(drop=True, level=levels_with_unique_vals)

    def fetch_one_result(self, iloc: Optional[int] = None, **kwargs):
        if len(kwargs) > 0:
            results_df_slice = self.results_df.xs(
                kwargs.values(), level=tuple(kwargs.keys()), drop_level=False
            )
            if len(results_df_slice) == 0:
                raise ValueError(f"No results found for {kwargs}")
        else:
            results_df_slice = self.results_df

        assert isinstance(results_df_slice, pd.Series)

        if iloc is not None:
            results_df_filename = results_df_slice.iloc[iloc]
        else:
            if len(results_df_slice) > 1:
                remaining_results = self._minify_df(results_df_slice)
                if len(remaining_results.index.names) == 1 and remaining_results.index.names[0] is None:
                    print("duplicated results found, selecting the last one")
                else:
                    ambiguous_index_vals = [
                        remaining_results.to_frame("_discard").reset_index().loc[:, n].unique()
                        for n in remaining_results.index.names
                    ]
                    raise ValueError(
                        f"More than one result found for the given kwargs. "
                        f"The ambiguous levels are: {remaining_results.index.names}. ",
                        f"The ambiguous values are: {ambiguous_index_vals}",
                    )
            results_df_filename = results_df_slice.iloc[-1]

        with open(self.abs_folder_name / results_df_filename, "rb") as f:
            ret = pickle.load(f)
            if len(ret) == 3:
                idxs, res, slurm_metadata = ret
            elif len(ret) == 2:
                idxs, res = ret
                slurm_metadata = SLURMJobMetadata(None, None, False)
            else:
                raise ValueError(f"Unexpected number of elements in pickle file: {len(ret)}")

        assert isinstance(idxs, dict)

        return Result(
            res,
            metadata_kwargs={
                k: v
                for k, v in idxs.items()
                if k in self.minimal_results_df.index.names
            },
            all_kwargs=idxs,
            slurm_metadata=slurm_metadata,
        )


    def refresh_results_db(self, func, **kwargs):
        return refetch_results_db(self.abs_folder_name, func, **kwargs)


    def fetch_evaluation_results(self, **kwargs):
        if len(kwargs) > 0:
            results_df_slice = self.results_df.xs(
                kwargs.values(), level=tuple(kwargs.keys()), drop_level=False
            )
        else:
            results_df_slice = self.results_df

        assert isinstance(results_df_slice, pd.Series)

        results = []

        def _get_single_result(path):
            with open(path, "rb") as f:
                res = pickle.load(f)
            if res[1].eval_results is not None:
                return {"mmd": res[1].eval_results.mmd, "c2st": res[1].eval_results.c2st}
            else:
                return {"mmd": None, "c2st": None}

        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(_get_single_result)(self.abs_folder_name / filename)
            for _, filename in results_df_slice.iteritems()
        )
        return pd.DataFrame(results, index=results_df_slice.index)
