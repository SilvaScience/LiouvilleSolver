"""Generic parallel helpers for SolverV9."""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def effective_n_jobs(n_jobs):
    """Normalize joblib-style worker counts."""
    cpu_count = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        return max(1, cpu_count + 1 + n_jobs)
    return max(1, n_jobs)


def make_index_blocks(n_items, n_jobs, block_size=None, configured_block_size=None):
    """Split integer indices into contiguous work blocks."""
    n_items = int(n_items)
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")
    if n_items == 0:
        return []
    if block_size is None:
        block_size = configured_block_size
    if block_size is None:
        target_blocks = max(1, 4 * max(1, int(n_jobs)))
        block_size = max(1, math.ceil(n_items / target_blocks))
    block_size = max(1, int(block_size))
    return [
        list(range(start, min(start + block_size, n_items)))
        for start in range(0, n_items, block_size)
    ]


def parallel_context(blas_threads, configured_blas_threads=None):
    """Limit nested BLAS/OpenMP threads while an outer loop runs."""
    if blas_threads is None:
        blas_threads = configured_blas_threads
    if blas_threads is None or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=int(blas_threads))


def run_blocks(blocks, worker, parallel_backend="serial", n_jobs=1):
    """Run block workers using a generic serial or threading backend."""
    if parallel_backend is None:
        parallel_backend = "serial"
    parallel_backend = str(parallel_backend).lower()
    if parallel_backend in {"serial", "none"} or int(n_jobs) == 1:
        return [item for block in blocks for item in worker(block)]
    if parallel_backend == "threading":
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            nested = list(executor.map(worker, blocks))
        return [item for block_result in nested for item in block_result]
    raise ValueError("parallel_backend must be 'serial' or 'threading'.")
