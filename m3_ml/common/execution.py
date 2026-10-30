"""CPU-budgeted process jobs with ordered results and atomic numeric checkpoints."""
import hashlib
import importlib.metadata
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .compute import cpu_threads

ROOT = Path(__file__).resolve().parents[2]
THREAD_ENV = ('M3_CPU_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _json(value):
    return json.dumps(value, sort_keys=True, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))


class ValidationCache:
    """A new data/code/library/resource signature gets a separate cache directory."""
    def __init__(self, name, inputs, root=None):
        digest = hashlib.sha256()
        for key, value in sorted(inputs.items()):
            digest.update(key.encode())
            if isinstance(value, (pd.DataFrame, pd.Series)):
                digest.update(repr(getattr(value, 'columns', value.name if isinstance(value, pd.Series) else None)).encode())
                digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
            else:
                digest.update(_json(value).encode())
        files = [ROOT / 'data/processed/wti_publication.csv', ROOT / 'm3_ml/tabular/models.py']
        for directory in ('common', 'distribution', 'risk', 'residual'):
            files.extend(sorted((ROOT / 'm3_ml' / directory).glob('*.py')))
        for path in files:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        versions = {package: importlib.metadata.version(package) for package in
                    ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'arch', 'quantile-forest')}
        digest.update(_json(versions).encode())
        digest.update(_json({'threads': cpu_threads(), 'workers': os.environ.get('M3_OUTER_JOBS', '1'),
                             'device': os.environ.get('M3_LIGHTGBM_DEVICE', 'cpu')}).encode())
        self.path = (Path(root) if root is not None else ROOT / '.local/frozen_validation_cache') / name / digest.hexdigest()
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / 'manifest.json'
        if not manifest.exists():
            manifest.write_text(_json({'name': name, 'signature': self.path.name, 'versions': versions}) + '\n')

    def file(self, key):
        return self.path / (hashlib.sha256(_json(key).encode()).hexdigest() + '.npz')

    def get(self, key):
        path = self.file(key)
        if not path.exists():
            return False, None
        with np.load(path, allow_pickle=False) as data:
            result = data['value']
        return True, result.item() if result.ndim == 0 else result

    def put(self, key, value):
        path = self.file(key)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, value=np.asarray(value))
        temporary.replace(path)


def input_frames(inp):
    return {'table': inp.table, **{f'features/{name}': frame for name, frame in inp.features.items()}}


def _initialize(function, context, threads):
    global _FUNCTION, _CONTEXT, _LIMIT
    for name in THREAD_ENV:
        os.environ[name] = str(threads)
    os.environ['M3_OUTER_JOBS'] = '1'
    _FUNCTION, _CONTEXT = function, context
    _LIMIT = threadpool_limits(limits=threads)


def _execute(task):
    return _FUNCTION(_CONTEXT, task)


def run_jobs(function, tasks, context, label, cache=None, cache_key=None):
    """Parallelize independent jobs only; collect by index to preserve selection ties."""
    tasks = list(tasks)
    total_threads = cpu_threads()
    requested = os.environ.get('M3_OUTER_JOBS', '1')
    workers = total_threads if requested == 'all' else int(requested)
    if not 1 <= workers <= total_threads:
        raise ValueError('outer workers must fit within M3_CPU_THREADS')
    workers = min(workers, max(1, len(tasks)))
    inner = max(1, total_threads // workers)
    results, pending_jobs = [None] * len(tasks), []
    for index, task in enumerate(tasks):
        key = cache_key(task) if cache_key else task
        hit, value = cache.get(key) if cache else (False, None)
        if hit:
            results[index] = value
        else:
            pending_jobs.append((index, task, key))
    complete = len(tasks) - len(pending_jobs)
    print(f'{label}: {len(tasks)} jobs; cached={complete}; workers={workers}; '
          f'threads/worker={inner}', flush=True)

    def record(index, task, key, value):
        nonlocal complete
        if cache:
            cache.put(key, value)
        results[index] = value
        complete += 1
        print(f'{label}: completed={complete}/{len(tasks)} job={index + 1} task={task}', flush=True)

    if workers == 1:
        with threadpool_limits(limits=inner):
            for index, task, key in pending_jobs:
                record(index, task, key, function(context, task))
        return results
    if not pending_jobs:
        return results
    previous = {name: os.environ.get(name) for name in THREAD_ENV}
    for name in THREAD_ENV:
        os.environ[name] = str(inner)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                               initargs=(function, context, inner))
    try:
        iterator = iter(pending_jobs)
        futures = {}

        def submit_next():
            item = next(iterator, None)
            if item is not None:
                futures[pool.submit(_execute, item[1])] = item

        for _ in range(workers * 2):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index, task, key = futures.pop(future)
                record(index, task, key, future.result())
                submit_next()
    except BaseException:
        if hasattr(pool, 'terminate_workers'):
            pool.terminate_workers()
        else:
            pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return results
