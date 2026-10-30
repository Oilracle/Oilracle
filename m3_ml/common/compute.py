"""Bounded classical-model execution; statistical settings stay in each model."""
import os
from functools import wraps

from threadpoolctl import threadpool_limits


def cpu_threads():
    """Explicit budget; 'all' uses the host allocation authorized by the user."""
    raw = os.environ.get('M3_CPU_THREADS', '2')
    available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    if raw.lower() == 'all':
        return available
    try:
        threads = int(raw)
    except ValueError as exc:
        raise ValueError('M3_CPU_THREADS must be all or a positive integer within the CPU allocation') from exc
    if not 1 <= threads <= available:
        raise ValueError(f'M3_CPU_THREADS must be between 1 and {available}')
    return threads


def lightgbm_params():
    """Select CPU or OpenCL GPU explicitly; failed GPU fits must propagate."""
    device = os.environ.get('M3_LIGHTGBM_DEVICE', 'cpu').lower()
    if device not in ('cpu', 'gpu'):
        raise ValueError('M3_LIGHTGBM_DEVICE must be cpu or gpu')
    return {'device_type': device, 'n_jobs': cpu_threads()}


def bounded_cpu(function):
    """Bound BLAS/OpenMP as well as the estimators' explicit thread budgets."""
    @wraps(function)
    def run(*args, **kwargs):
        with threadpool_limits(limits=cpu_threads()):
            return function(*args, **kwargs)
    return run


def report_compute():
    params = lightgbm_params()
    workers = os.environ.get('M3_OUTER_JOBS', '1')
    print(f'compute: outer jobs={workers}; total CPU budget={params["n_jobs"]}; '
          f'LightGBM device={params["device_type"]} (no fallback)', flush=True)
