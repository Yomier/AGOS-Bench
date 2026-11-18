"""Compute the nine manuscript metrics from a run's locked episode results."""
import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from agos.eval.metric_contract import (
    METRIC_REVISION,
    aggregate_navigation_error,
    aggregate_osr,
    aggregate_verification_actions,
    assert_uniform_metric_protocol,
)
from agos.eval.metric_primitives import compute_ase
from benchmark.scoring import classify, read_terminal, vision_counts


def percent(value):
    return None if value is None else 100 * value


def unit_mean(rows, field):
    values = [row.get(field) for row in rows]
    if values and all(type(v) in (float, int) and isfinite(v) and 0 <= v <= 1 for v in values):
        return fmean(values)
    return None


def aggregate_paper_metrics(episodes, *, model_decisions):
    """Aggregate adjudicated episodes. Normalized metrics use the manuscript's percent scale."""
    if episodes and assert_uniform_metric_protocol(episodes) != [METRIC_REVISION]:
        raise ValueError('The selected results do not use the current metric protocol.')
    for row in episodes:
        protocol = row.get('success_protocol') or {}
        if (protocol.get('id') != 'AGOS_CONFIRM30_V1' or protocol.get('max_confirm_distance_m') != 30
                or protocol.get('require_los') is not True or protocol.get('require_gt_visible_in_stop_6view') is not True
                or protocol.get('require_stop_observe') is not False or type(row.get('success')) is not bool):
            raise ValueError('Expected a boolean outcome and the Confirm30 success protocol.')
    aerial = [row for row in episodes if row.get('aerial_metrics_applicable', True)]
    spl_valid = all(row.get('spl_valid') is True and row.get('spl_oracle_version') == 'semantic_meters_confirm_layered_v3' for row in episodes)
    ase_valid = all(row.get('ase_sampling_valid') is True and row.get('ase_max_uav_decision_steps') == 80
                    and 'ase_first_oracle_visible_uav_step' in row for row in aerial)
    successes = sum(row['success'] for row in episodes)
    osr = aggregate_osr(episodes)
    gva = aggregate_verification_actions(episodes)
    ne = aggregate_navigation_error(episodes, current=True)
    calls = [row.get('vlm_calls') for row in episodes]
    valid_calls = bool(episodes) and model_decisions
    for row, count in zip(episodes, calls):
        by_role = [row.get('uav_vlm_calls'), row.get('ugv_vlm_calls')]
        if (type(count) is not int or count < 0 or (row['success'] and count == 0)
                or any(type(n) is not int or n < 0 for n in by_role) or count != sum(by_role)):
            valid_calls = False
    metrics = {
        'SR': percent(successes / len(episodes)) if episodes else None,
        'OSR': percent(osr['OSR']),
        'SPL': percent(unit_mean(episodes, 'UGV_SPL_contribution')) if spl_valid else None,
        'NE': ne['avg_NE_m'],
        'ASE': percent(compute_ase(aerial)) if ase_valid else None,
        'ACG': percent(unit_mean(aerial, 'ACG_episode')),
        'GVA': percent(gva['GVA']),
        'DS': fmean(calls) if valid_calls else None,
        'DSR': percent(fmean(row['success'] / max(count, 1) for row, count in zip(episodes, calls))) if valid_calls else None,
    }
    return {'metrics': metrics, 'scored': len(episodes), 'successes': successes,
            'support': {'aerial_episodes': len(aerial), 'navigation_poses': ne['NE_valid_samples'],
                        'verification_events': gva['gva_support'], 'verification_episodes': gva['gva_episode_support'],
                        **{name: gva['gva_' + name] for name in ('tp', 'tn', 'fp', 'fn')}}}


def score_run(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    episodes_dir = (run_dir / 'episodes').resolve()
    pending, rows, seen = [], [], set()
    for task in manifest['tasks']:
        key = task['key']
        folder = (episodes_dir / key).resolve()
        segments = key.split('/')
        if (len(segments) != 3 or any(not part or part in ('.', '..') or '\\' in part for part in segments)
                or key in seen or not folder.is_relative_to(episodes_dir)):
            raise ValueError(f'Invalid or duplicate manifest task: {key}')
        seen.add(key)
        terminal = read_terminal(folder)
        if terminal is None or terminal['outcome'] == 'pending':
            pending.append(key)
            continue
        result_path = (folder / terminal['attempt'] / terminal['result']).resolve()
        if not result_path.is_relative_to(folder):
            raise ValueError(f'Invalid result path for {key}')
        row = json.loads(result_path.read_text(encoding='utf-8'))
        if row.get('release_task_key') != key or row.get('release_plan_sha256') != task['plan_sha256']:
            raise ValueError(f'Locked task identity disagrees with manifest for {key}')
        vision_errors, vision_ok = vision_counts(result_path.parent / 'ag_os_trace.jsonl')
        if (vision_errors != terminal['vision_errors'] or vision_ok != terminal['vision_ok']
                or classify(row, vision_errors, vision_ok) != terminal['outcome']):
            raise ValueError(f'Locked outcome disagrees with evidence for {key}')
        if manifest['arm'] == 'menu_oracle' and (row.get('vlm_calls') != 0 or row.get('http_attempts', 0) != 0):
            raise ValueError(f'Oracle result includes model calls for {key}')
        rows.append(row)
    report = aggregate_paper_metrics(rows, model_decisions=manifest['arm'] != 'menu_oracle')
    return {**report, 'requested': len(seen), 'pending': pending, 'complete': bool(seen) and not pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Directory containing manifest.json and episodes/')
    args = parser.parse_args()
    report = score_run(args.run_dir)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
