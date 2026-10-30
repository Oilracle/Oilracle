"""Run remaining frozen-October experiments sequentially, stopping on failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def module(name: str, *args: str) -> list[str]:
    return ['uv', 'run', 'python', '-m', name, *args]


STEPS = [
    ('kronos_validate', module('m3_ml.kronos.run', 'validate', '--device', 'cuda', '--batch-sequences', '200')),
    ('kronos_final_train', module('m3_ml.kronos.train', '--stage', '2')),
    ('kronos_test', module('m3_ml.kronos.run', 'test', '--device', 'cuda', '--batch-sequences', '200')),
    ('kronos_forecast', module('m3_ml.kronos.run', 'forecast', '--device', 'cuda', '--batch-sequences', '200')),
    ('tabular', module('m3_ml.tabular.run_direct')),
    ('tabular_denoised', module('m3_ml.tabular.run_direct', '--denoise')),
    ('tabular_news', module('m3_ml.tabular.run_direct', '--news')),
    ('tabular_denoised_news', module('m3_ml.tabular.run_direct', '--denoise', '--news')),
    ('tabular_analyze', module('m3_ml.tabular.analyze_direct')),
    ('distribution_validate', module('m3_ml.distribution.run', 'validate')),
    ('distribution_test', module('m3_ml.distribution.run', 'test')),
    ('distribution_analyze', module('m3_ml.distribution.analyze')),
    ('distribution_validation_report', module('m3_ml.distribution.report_validation')),
    ('volatility_validate', module('m3_ml.risk.volatility', 'validate')),
    ('events_validate', module('m3_ml.risk.events', 'validate')),
    ('volatility_test', module('m3_ml.risk.volatility', 'test')),
    ('events_test', module('m3_ml.risk.events', 'test')),
    ('risk_analyze', module('m3_ml.risk.analyze')),
    ('residual', module('m3_ml.residual.run')),
    ('residual_analyze', module('m3_ml.residual.analyze')),
    ('compare_classical', module('m3_ml.compare.part1')),
    ('compare_lstm', module('m3_ml.compare.lstm_aligned')),
    ('compare_analyze', module('m3_ml.compare.analyze')),
    ('video_data', ['uv', 'run', 'python', 'm4_reports/video/prepare.py']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-step', choices=[name for name, _ in STEPS], default=STEPS[0][0])
    parser.add_argument('--through-step', choices=[name for name, _ in STEPS],
                        help='Stop successfully after this step; later steps remain pending.')
    parser.add_argument('--list', action='store_true', help='Print the serial queue without running it.')
    parser.add_argument('--cpu-threads', default='2', help='Total CPU budget, or all.')
    parser.add_argument('--workers', default='1', help='Independent validation workers, or all.')
    args = parser.parse_args()
    start = next(i for i, (name, _) in enumerate(STEPS) if name == args.from_step)
    stop = next(i + 1 for i, (name, _) in enumerate(STEPS) if name == args.through_step) if args.through_step else len(STEPS)
    if stop <= start:
        parser.error('--through-step must not precede --from-step')
    if args.list:
        for name, command in STEPS[start:stop]:
            print(name, subprocess.list2cmdline(command))
        return
    env = os.environ.copy()
    available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    threads = available if args.cpu_threads == 'all' else int(args.cpu_threads)
    workers = threads if args.workers == 'all' else int(args.workers)
    if not 1 <= workers <= threads <= available:
        parser.error('require 1 <= workers <= CPU threads <= available CPUs')
    env.update({key: str(threads) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                   'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'M3_CPU_THREADS')})
    env['M3_OUTER_JOBS'] = str(workers)
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('M3_LIGHTGBM_DEVICE', 'cpu')
    status = ROOT / '.local' / 'frozen_serial_status.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    # Windows hybrid-CPU scheduling can confine all background workers to the
    # efficiency cores. For the explicitly requested full-core run, keep one
    # independent worker per allowed core across successive validation pools.
    if os.name == 'nt' and workers == threads == available:
        with (status.parent / 'full_cpu_affinity.log').open('a') as affinity_log:
            subprocess.Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                              str(ROOT / 'm3_ml/pin_workers.ps1'), '-RunnerId',
                              str(os.getpid())], cwd=ROOT, stdout=affinity_log,
                             stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    if any(name in {'tabular_news', 'tabular_denoised_news', 'distribution_validate', 'distribution_test'}
           for name, _ in STEPS[start:stop]):
        from m3_ml.common.data import load_table
        from m3_ml.common.news import news_features, news_metadata, require_complete_features
        from m3_ml.common.walkforward import forecast_origins
        table = load_table()
        news_metadata()
        require_complete_features(news_features(table.date), table.date,
                                  forecast_origins(table.date, 'daily_update'))
        print('News release-calendar preflight passed for every October origin.', flush=True)
    for index, (name, command) in enumerate(STEPS[start:stop], start=start + 1):
        record = dict(step=name, index=index, total=len(STEPS), command=command,
                      runner_pid=os.getpid(), started=time.time(), status='running',
                      cpu_threads=threads, workers=workers)
        status.write_text(json.dumps(record, indent=2) + '\n')
        with (status.parent / 'frozen_serial_history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(record) + '\n')
        print(f'\n[{index}/{len(STEPS)}] START {name}: {subprocess.list2cmdline(command)}', flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        outcome = 'completed' if result.returncode == 0 else 'failed'
        record.update(finished=time.time(), returncode=result.returncode, status=outcome)
        status.write_text(json.dumps(record, indent=2) + '\n')
        with (status.parent / 'frozen_serial_history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(record) + '\n')
        print(f'[{index}/{len(STEPS)}] {outcome.upper()} {name}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f'Requested serial steps {start + 1}..{stop} completed successfully; '
          f'{len(STEPS) - stop} later steps remain.', flush=True)


if __name__ == '__main__':
    main()
