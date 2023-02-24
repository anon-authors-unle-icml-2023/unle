import glob
import os
import subprocess
import sys
import time
from asyncio.exceptions import CancelledError
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from distributed.client import as_completed
from pyloric import create_prior, simulate, summary_stats
from sbibm.tasks import Task
from sbibm.tasks.simulator import Simulator

global nan_replace_glob
global summary_glob
global NAMES
global CACHE

summary_glob = "summary_statistics"
nan_replace_glob = -99
NAMES_sim = [
    [
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "AB/PD",
        "LP",
        "LP",
        "LP",
        "LP",
        "LP",
        "LP",
        "LP",
        "LP",
        "PY",
        "PY",
        "PY",
        "PY",
        "PY",
        "PY",
        "PY",
        "PY",
        "Synapses",
        "Synapses",
        "Synapses",
        "Synapses",
        "Synapses",
        "Synapses",
        "Synapses",
    ],
    [
        "Na",
        "CaT",
        "CaS",
        "A",
        "KCa",
        "Kd",
        "H",
        "Leak",
        "Na",
        "CaT",
        "CaS",
        "A",
        "KCa",
        "Kd",
        "H",
        "Leak",
        "Na",
        "CaT",
        "CaS",
        "A",
        "KCa",
        "Kd",
        "H",
        "Leak",
        "AB-LP",
        "PD-LP",
        "AB-PY",
        "PD-PY",
        "LP-PD",
        "LP-PY",
        "PY-LP",
    ],
]
NAMES = [
    "AB/PD_Na",
    "AB/PD_CaT",
    "AB/PD_CaS",
    "AB/PD_A",
    "AB/PDK_Ca",
    "AB/PD_Kd",
    "AB/PD_H",
    "AB/PD_Leak",
    "LP_Na",
    "LP_CaT",
    "LP_CaS",
    "LP_A",
    "LP_KCa",
    "LP_Kd",
    "LP_H",
    "LP_Leak",
    "PY_Na",
    "PY_CaT",
    "PY_CaS",
    "PY_A",
    "PY_KCa",
    "PY_Kd",
    "PY_H",
    "PY_Leak",
    "SynapsesAB-LP",
    "SynapsesPD-LP",
    "SynapsesAB-PY",
    "SynapsesPD-PY",
    "SynapsesLP-PD",
    "SynapsesLP-PY",
    "SynapsesPY-LP",
]
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def seed_everything(seed):
    import random as python_random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    np.random.seed(seed)
    python_random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

from joblib import Memory


def mp_simulator(parameter):
    seed = int(parameter[-1])
    parameter = parameter[:-1]

    sample = pd.DataFrame(parameter.reshape(1, -1).numpy(), columns=NAMES_sim)
    x = simulate(sample.loc[0], seed=seed)

    if summary_glob == "summary_statistics":
        x = summary_stats(x).to_numpy()
    else:
        x = x["voltage"].flatten()
    x[np.isnan(x)] = nan_replace_glob

    return torch.tensor(x).float()


def slurm_simulator(thetas, simulation_batches=500):
    N = thetas.shape[0]
    if N < simulation_batches:
        simulation_batches = N
    jobs = N // simulation_batches

    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"seed_{j}.pkl"])

    for fl in glob.glob(DIR_PATH + os.sep + "slurm-*"):
        os.remove(fl)

    # Run the slurm jobs...
    for j in range(jobs):
        torch.save(
            thetas[j * simulation_batches : (j + 1) * simulation_batches, :],
            DIR_PATH + os.sep + f"thetas_{j}.pkl",
        )

    # Wait to for saving thetas
    time.sleep(10)

    for j in range(jobs):
        subprocess.run(
            [
                "sbatch",
                DIR_PATH + os.sep + "run_one.sh",
                DIR_PATH + os.sep + f"thetas_{j}.pkl",
                f"{j}",
                DIR_PATH,
                f"--output={DIR_PATH}",
            ]
        )

    time.sleep(10)

    # Check for complettion
    start_time = time.time()
    jobs_status = np.zeros(jobs)
    i = 0
    while True:
        if (i % 1000) == 0:
            for j in range(jobs):
                jobs_status[j] = os.path.isfile(DIR_PATH + os.sep + f"xs_{j}.pkl")
            sys.stdout.write(f"\rCompleted {int(jobs_status.sum())}/{jobs} jobs")
            sys.stdout.flush()
            if jobs_status.sum() == jobs:
                break
            current_time = time.time()
            time_till_execution = current_time - start_time
            if time_till_execution > 300:
                start_time = time.time()
                for j in range(jobs):
                    if not jobs_status[j]:
                        subprocess.run(
                            [
                                "sbatch",
                                DIR_PATH + os.sep + "run_one.sh",
                                DIR_PATH + os.sep + f"thetas_{j}.pkl",
                                f"{j}",
                                DIR_PATH,
                                f"--output={DIR_PATH}",
                            ]
                        )

        i += 1

    # Wait to receive xs
    time.sleep(10)
    subprocess.run(["scancel", "-n", "run_one.sh"])

    # if jobs_status.sum() != jobs:
    #     return slurm_simulator(thetas)
    # Append final results
    xs = []
    for j in range(jobs):
        xs.append(torch.load(DIR_PATH + os.sep + f"xs_{j}.pkl"))

    x = torch.vstack(xs)

    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep + f"seed_{j}.pkl"])

    for fl in glob.glob(DIR_PATH + os.sep + "slurm-*"):
        os.remove(fl)
    return x.float()


def worker_mp_simulator(parameters, num_cores):
    # hijack dask multiprocessing parallelism for finer batching strategies.
    NUM_SAMPLES = parameters.shape[0]
    seed = torch.randint(0, 2 ** 32 - 1, (NUM_SAMPLES, 1)).float()
    paras = torch.hstack((parameters, seed))
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    # e = ProcessPoolExecutor(max_workers=num_cores)
    # e = ThreadPoolExecutor(max_workers=num_cores)
    from loky import get_reusable_executor

    e = get_reusable_executor(max_workers=num_cores, timeout=2)
    xs = e.map(mp_simulator, paras)
    xs = torch.vstack(list(xs))
    return xs


# @m.cache
def _simulate_one_sample(param, seed):
    x = simulate(param, seed=seed)  # type: ignore
    if summary_glob == "summary_statistics":
        x = summary_stats(x).to_numpy(dtype='float32')  # type: ignore
        assert len(x.shape) == 2
    else:
        x = x["voltage"].flatten()  # type: ignore
    return x

def worker_sequential_simulator(parameters, seeds):
    num_samples = parameters.shape[0]
    xs = []
    # seed = torch.randint(0,2**32-1, (num_samples))
    import numpy as np
    for i in range(num_samples):
        sample = pd.DataFrame(parameters[i].reshape(1, -1).numpy(), columns=NAMES_sim)
        x = _simulate_one_sample(sample.loc[0], seed=seeds[i])

        # x[np.isnan(x)] = nan_replace_glob
        xs.append(x)

    xs = np.concatenate(xs, axis=0, dtype=np.float32)
    ret = torch.from_numpy(xs)
    return ret


class Pyloric(Task):
    def __init__(self, summary="summary_statistics", nan_replace=-99):
        self.summary = summary
        self.nan_replace = nan_replace
        self.dim_data_unflatten = torch.Size((3, 440000))
        self.dim_data_raw = torch.numel(torch.tensor(self.dim_data_unflatten))
        if summary == "summary_statistics":
            dim_data = 15
        else:
            dim_data = self.dim_data_raw

        observation_seeds = [4933, "na", 42]

        super().__init__(
            dim_parameters=31,
            dim_data=dim_data,
            name="pyloric",
            name_display="Pyloric STG",
            num_observations=[1],  # pyright: ignore [reportGeneralTypeIssues]
            observation_seeds=observation_seeds,
            num_posterior_samples=10000,  # pyright: ignore [reportGeneralTypeIssues]
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior = create_prior()
        from pyro.distributions import Uniform, Independent
        self.prior.numerical_prior = Independent(
            Uniform(
                low=self.prior.numerical_prior.base_dist.low,
                high=self.prior.numerical_prior.base_dist.high
            ),
            reinterpreted_batch_ndims=1
        )

        self.prior_dist = self.prior.numerical_prior
        self.t = torch.arange(0, 11000, 0.025)
        self.names = NAMES

    def get_prior(self):
        def prior(num_samples=1):
            return self.prior.sample((num_samples,))

        return prior

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations"""
        if self.summary is None:
            return data.reshape(-1, *self.dim_data_unflatten)
        else:
            return data.reshape(-1, self.dim_data)

    def get_simulator(
        self,
        max_calls=None,
        nan_replace=0.0,
        seed=None,
        sim_type="dask_local",
        num_cores=100,
        save_simulations=True,
        job_extra_args: tuple = (),
        verbose=1
    ):
        if sim_type == "sequential":

            def sequential_simulator(parameters):
                num_samples = parameters.shape[0]
                xs = []
                for i in range(num_samples):
                    sample = pd.DataFrame(
                        parameters[i].reshape(1, -1).numpy(), columns=self.prior.names
                    )
                    x = simulate(sample.loc[0], seed=seed)
                    if self.summary == "summary_statistics":
                        x = summary_stats(x).to_numpy()
                    else:
                        x = x["voltage"].flatten()

                    x[np.isnan(x)] = nan_replace
                    xs.append(torch.tensor(x))
                return torch.vstack(xs).float()

            return Simulator(
                task=self, simulator=sequential_simulator, max_calls=max_calls
            )
        elif sim_type in ("dask", "dask_local"):
            if seed is None:
                random_seed = 0
            else:
                assert isinstance(seed, int)
                random_seed = seed

            def simulator(parameters):
                num_samples = parameters.shape[0]

                from distributed.client import Client

                scheduler_addr = os.environ.get('PYLORIC_DASK_CLUSTER_ADDR')
                if scheduler_addr is not None:
                    if verbose:
                        print(f'connecting to existing cluster at: {scheduler_addr}')
                    cluster = None
                    client = Client(scheduler_addr, timeout=30.0)  # pyright: ignore [reportGeneralTypeIssues]
                    close_after_simulation = False

                    num_workers = len(client._scheduler_identity['workers'])
                    batch_size = min(num_samples // num_workers, 50)
                else:
                    if verbose:
                        print('creating dask cluster...', end='')
                    from dask_jobqueue import SLURMCluster
                    close_after_simulation = True
                    from pathlib import Path
                    Path("dask-simulation-logs").mkdir(exist_ok=True)
                    if sim_type == "dask_local":
                        print("(local)")
                        from dask.distributed import LocalCluster
                        import joblib
                        import dask 
                        num_workers = joblib.cpu_count()
                        cluster = LocalCluster(
                            n_workers=num_workers,
                            threads_per_worker=1,
                            processes=True,
                        )
                    else:
                        cluster = SLURMCluster(  # type: ignore
                            n_workers=0,  # create the workers "lazily" (upon cluster.scale)
                            memory="4GB",  # amount of RAM per worker
                            processes=1,  # number of execution units per worker (threads and processes)
                            cores=1,  # among those execution units, number of processes
                            job_extra=[
                                # '--export=ALL', # default behavior.
                                "--output=dask-simulation-logs/R-%x.%j.out",
                                "--error=dask-simulation-logs/R-%x.%j.err",
                                *job_extra_args,
                                # '--export=OMP_NUM_THREADS=1,MKL_NUM_THREADS=1',
                            ],
                            # extra = ["--no-nanny"],
                            # scheduler_options={'dashboard_address': ':8787', 'port': 45987, 'allowed_failures': 10},
                            scheduler_options={
                                "dashboard_address": ":8787",
                                "allowed_failures": 10,
                            },
                            job_cpu=1,
                            walltime="2:0:0",
                        )
                        if verbose:
                            print(cluster.job_script())
                    # cluster.adapt(minimum=0, maximum=100)

                        num_workers = min(num_samples, num_cores)

                    if verbose:
                        print(f"scaling to {num_workers} workers...", end='')
                    cluster.scale(num_workers)
                    if verbose:
                        print(' OK.')

                    if verbose:
                        print("Connecting to cluster...", end="", flush=True)
                    # client = Client("tcp://127.0.0.1:45987", timeout=30.)
                    client = Client(cluster, timeout=30.0)  # pyright: ignore [reportGeneralTypeIssues]
                    if verbose:
                        print(" OK.", flush=True)

                    batch_size = min(num_samples // num_workers, 50)

                if verbose:
                    print("testing dask cluster...", end="", flush=True)
                client.submit(lambda x: x, "hello", priority=10).result()

                if verbose:
                    print(" OK.", flush=True)

                    print(f"using {batch_size=}")

                futs = []
                xs = torch.zeros((num_samples, self.dim_data), dtype=torch.float32)

                def simulate_and_get_idx(i, parameters, slice, seeds):
                    return worker_sequential_simulator(parameters[slice], seeds[slice]), i

                from tqdm import tqdm
                from tqdm.auto import tqdm

                if verbose:
                    print("scattering data to workers...", end="")
                num_batches = num_samples // batch_size + 1 * (
                    (num_samples % batch_size) > 0
                )


                seeds = np.random.RandomState(random_seed).randint(0, 2**32 - 1, size=(num_samples,))

                [fut_params, fut_seeds] = client.scatter([parameters, seeds], broadcast=True) # type: ignore
                if verbose:
                    print(" OK.")

                try:
                    if verbose:
                        print("submitting batches...")
                    for i in tqdm(range(num_batches)):
                        slice_ = slice(
                            i * batch_size, min((i + 1) * batch_size, num_samples)
                        )
                        f = client.submit(
                            simulate_and_get_idx,
                            i,
                            fut_params,
                            slice_,
                            fut_seeds,
                            retries=5,
                        )
                        futs.append(f)

                    # for _, ret in tqdm(as_completed(futs, with_results=True)):
                    if verbose:
                        print("OK")
                        print("waiting for results...")
                    num_valid = 0
                    total_done = 0
                    pbar = tqdm(as_completed(futs, with_results=True), total=num_batches)
                    for _, ret in pbar:
                        # for _, ret in as_completed(futs, with_results=True):
                        x, i = ret  # type: ignore
                        this_round_num_valid = cast(torch.Tensor, x).isfinite().all(1).sum()
                        num_valid += this_round_num_valid
                        total_done += len(x)
                        pbar.set_postfix_str(f"{num_valid}/{total_done} valid simulations")

                        # for some reason, this segfaults
                        # xs[i * batch_size : min((i + 1) * batch_size, num_samples)] = x
                        xs[i * batch_size : min((i + 1) * batch_size, num_samples)] = x.double().float()
                    if verbose:
                        print("OK.")
                finally:
                    if close_after_simulation:
                        client.close()
                        assert cluster is not None
                        cluster.close()
                # return xs.float()
                return xs

            return Simulator(task=self, simulator=simulator, max_calls=max_calls)
        elif sim_type == "parallel":

            def simulator(parameters):
                NUM_SAMPLES = parameters.shape[0]
                seed = torch.randint(0, 2 ** 32 - 1, (NUM_SAMPLES, 1)).float()
                paras = torch.hstack((parameters, seed))
                with Pool(num_cores) as pool:
                    xs = pool.map(mp_simulator, paras)
                xs = torch.vstack(xs)
                return xs

            return Simulator(task=self, simulator=simulator, max_calls=max_calls)
        elif sim_type == "slurm":
            return Simulator(task=self, simulator=slurm_simulator, max_calls=max_calls)
        else:
            raise NotImplementedError()

    def get_precomputed_dataset(self):

        thetas = None
        xs = None
        for i in range(5):
            df_paras = pd.read_pickle(
                str(Path(__file__).parent.absolute())
                + f"/files/all_circuit_parameters_{i}.pkl"
            )
            df_simulation_output = pd.read_pickle(
                str(Path(__file__).parent.absolute())
                + f"/files/all_simulation_outputs_{i}.pkl"
            )
            if i == 0:
                thetas = torch.tensor(df_paras.to_numpy()).float()
                xs = torch.tensor(df_simulation_output.to_numpy()[:, :15]).float()
                xs[np.isnan(xs)] = self.nan_replace
            else:
                assert isinstance(thetas, torch.Tensor)
                assert isinstance(xs, torch.Tensor)
                thetas = torch.vstack(
                    [thetas, torch.tensor(df_paras.to_numpy()).float()]
                )
                xs = torch.vstack(
                    [xs, torch.tensor(df_simulation_output.to_numpy()[:, :15]).float()]
                )
                xs[np.isnan(xs)] = self.nan_replace
        return thetas, xs
