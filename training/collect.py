"""Collect new privileged expert traces on the Train1100 split using the benchmark runner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Root of unpacked task volumes')
    parser.add_argument('--config', type=Path, required=True, help='Resolved AGOS-Agent YAML used by the runner')
    parser.add_argument('--output', type=Path, required=True, help='New directory for config, captures and run')
    parser.add_argument('--episode', action='append', default=[], help='Town/episode key; repeat or omit for all Train1100')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--simulator-root', type=Path, default=os.environ.get('AGOS_SIM_ROOT'),
                        help='CarlaAir installation; defaults to AGOS_SIM_ROOT')
    parser.add_argument('--native-profile', type=Path, help='Build-specific native guard profile for the runner')
    parser.add_argument('--dry-run', action='store_true', help='Have the benchmark runner validate the plan offline')
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    coop = config['cooperative_search']
    if coop.get('arm') != 'agos_b':
        parser.error('Pass the agos_b AGOS-Agent configuration')
    if args.workers < 1:
        parser.error('workers must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    coop['expert'] = {'enabled': True, 'mode': 'drive', 'confirm_capture': True,
                      'confirm_distance_m': 29.0, 'annotate_proposal_max_m': 30.0,
                      'min_views': 1, 'view_margin': 0.9}
    coop['capture_root'] = str(output / 'captures')
    train_id = 'AGOS_PAPER_TRAIN1100_CARS34_V1_20261017'
    config.setdefault('benchmark', {})['dataset_id'] = train_id
    config.setdefault('paper_experiment', {})['dataset_id'] = train_id
    config_file = output / 'expert.yaml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    cmd = [sys.executable, '-m', 'benchmark.run_parallel_eval', '--data-root', str(args.data_root.resolve()),
           '--split', 'train1100', '--arm', 'agos_b', '--config', str(config_file),
           '--run-dir', str(output / 'run'), '--workers', str(args.workers)]
    for episode in args.episode:
        cmd.extend(['--episode', episode])
    if args.native_profile:
        cmd.extend(['--native-profile', str(args.native_profile.resolve())])
    if args.dry_run:
        cmd.append('--dry-run')
    env = os.environ.copy()
    if args.simulator_root:
        env['AGOS_SIM_ROOT'] = str(Path(args.simulator_root).resolve())
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
