"""Parallel execution helpers for SolverV8."""

import math
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


_PROCESS_SOLVER = None


def _init_process_solver(solver):
    """Store one solver copy per process worker."""
    global _PROCESS_SOLVER
    _PROCESS_SOLVER = solver


def _calc_w3_block_process(block, w_list, tau2, integration_weights, spectrum_components):
    """ProcessPool worker entry point using the process-local solver."""
    if _PROCESS_SOLVER is None:
        raise RuntimeError("Process worker was not initialized with a solver.")
    return _PROCESS_SOLVER._calc_w3_block(
        block, w_list, tau2, integration_weights, spectrum_components
    )


def _calc_w3_block_joblib(solver, block, w_list, tau2, integration_weights, spectrum_components):
    """Joblib worker entry point for process-style backends."""
    return solver._calc_w3_block(
        block, w_list, tau2, integration_weights, spectrum_components
    )


def effective_n_jobs(n_jobs):
    """Normalize joblib-style worker counts."""
    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    if n_jobs < 0:
        return max(1, cpu_count + 1 + n_jobs)
    return max(1, int(n_jobs))


def make_w3_blocks(n_w, n_jobs, block_size=None, configured_block_size=None):
    """Split omega_3 column indices into backend tasks."""
    if block_size is None:
        block_size = configured_block_size
    if block_size is None:
        target_blocks = max(1, 4 * max(1, n_jobs))
        block_size = max(1, math.ceil(n_w / target_blocks))
    block_size = max(1, int(block_size))
    return [
        list(range(start, min(start + block_size, n_w)))
        for start in range(0, n_w, block_size)
    ]


def parallel_context(blas_threads, configured_blas_threads=None):
    """Limit nested BLAS/OpenMP threads while an outer loop runs."""
    if blas_threads is None:
        blas_threads = configured_blas_threads
    if blas_threads is None or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=blas_threads)


def run_w3_blocks(
    solver,
    blocks,
    w_list,
    tau2,
    integration_weights,
    parallel_backend,
    n_jobs,
    spectrum_components,
):
    """Run omega_3 blocks using the selected parallel backend."""
    if parallel_backend in {"serial", None} or n_jobs == 1:
        return [
            item
            for block in blocks
            for item in solver._calc_w3_block(
                block, w_list, tau2, integration_weights, spectrum_components
            )
        ]

    if parallel_backend == "threading":
        nested = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(solver._calc_w3_block)(
                block, w_list, tau2, integration_weights, spectrum_components
            )
            for block in blocks
        )
    elif parallel_backend in {"loky", "multiprocessing"}:
        nested = Parallel(
            n_jobs=n_jobs,
            backend=parallel_backend,
            max_nbytes="10M",
            mmap_mode="r",
        )(
            delayed(_calc_w3_block_joblib)(
                solver,
                block,
                w_list,
                tau2,
                integration_weights,
                spectrum_components,
            )
            for block in blocks
        )
    elif parallel_backend in {"process", "processpool"}:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_process_solver,
            initargs=(solver,),
        ) as pool:
            futures = [
                pool.submit(
                    _calc_w3_block_process,
                    block,
                    w_list,
                    tau2,
                    integration_weights,
                    spectrum_components,
                )
                for block in blocks
            ]
            nested = [future.result() for future in futures]
    else:
        raise ValueError(
            "parallel_backend must be one of: "
            "'serial', 'threading', 'loky', 'multiprocessing', or 'process'"
        )

    return [item for block_result in nested for item in block_result]
