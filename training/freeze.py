"""Freeze completed Train1100 expert captures into offline multimodal trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .contracts import student_input, validate_target
from .corpus import new_directory, row_role
from .targets import canonical_target, surface_target, _legal
from .train import TrainError, validate_training_row

EXPERT_SITES = {'expert_decision', 'expert_verify', 'expert_confirm'}


def site_row(call, step, episode_key, output):
    kind = call['call_kind']
    public = student_input(call['public_input'], {})
    decision = step.get('expert_decision') or {}
    if decision.get('eligibility') == 'defer_no_public_region':
        return None
    target = call.get('target') or {}
    if kind == 'expert_confirm':
        target = {'target_verdict': target['target_verdict']}
    if not target:
        return None
    problems = validate_target(public, target)
    if problems:
        raise ValueError(f"{call['call_id']}: {', '.join(problems)}")
    flat, problem = surface_target({'expert_decision': decision, 'public_input': public}, canonical_target(target))
    if not flat:
        if problem:
            raise ValueError(f"{call['call_id']}: {problem}")
        return None
    if not _legal(public, flat):
        raise ValueError(f"{call['call_id']}: action outside the recorded menu")
    paths = call['image_paths']
    digests = call['image_sha256']
    if not paths or len(paths) != len(digests):
        raise ValueError(f"{call['call_id']}: incomplete image slots")
    images = []
    for path, digest in zip(paths, digests):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{call['call_id']}: image hash mismatch")
        name = f'images/{digest}.png'
        if not (output / name).exists():
            (output / name).write_bytes(data)
        images.append(name)
    text = call['user_prompt']
    if json.loads(text) != public:
        raise ValueError(f"{call['call_id']}: deployed prompt differs from the recorded public input")
    messages = [{'role': 'system', 'content': call['system_prompt']},
                {'role': 'user', 'content': [{'type': 'image', 'path': p} for p in images] +
                                            [{'type': 'text', 'text': text}]},
                {'role': 'assistant', 'content': json.dumps(flat, sort_keys=True, separators=(',', ':'))}]
    row = {'images': images, 'messages': messages,
           'meta': {'episode_key': episode_key, 'call_id': call['call_id'], 'role': step['role'],
                    'capture_site': kind, 'step': step['step']}}
    validate_training_row(row)
    row_role(row)
    return row


def freeze(run, destination):
    from benchmark.scoring import read_terminal

    tree = Path(run) / 'episodes' / 'train1100'
    terminals = sorted(tree.glob('*/*/terminal.json'))
    if not terminals:
        raise ValueError('No Train1100 terminal records found')
    counts = Counter()
    excluded = {}
    with new_directory(destination) as output:
        with (output / 'rows.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for terminal in terminals:
                key = terminal.parent.relative_to(tree).as_posix()
                selected = read_terminal(terminal.parent)
                if selected['outcome'] not in ('success', 'failure') or selected.get('vision_errors'):
                    excluded[key] = 'pending or visual fault'
                    continue
                attempt = terminal.parent / selected['attempt']
                result = json.loads((attempt / selected['result']).read_text(encoding='utf-8'))
                reason = result['failure_reason']
                faults = result.get('paper_fault_audit') or {}
                if (reason not in ('SUCCESS', 'BUDGET_EXHAUSTED', 'UGV_NAV_STALLED')
                        or result.get('infra_error_count') or result.get('false_confirm_count')
                        or result.get('would_false_confirm_count') or result.get('paper_formal_score_eligible') is False
                        or faults.get('events') or faults.get('failures')):
                    excluded[key] = reason
                    continue
                if result.get('episode_id') != terminal.parent.name:
                    raise ValueError(f'{key}: result identity mismatch')
                seen = set()
                with (attempt / 'ag_os_trace.jsonl').open(encoding='utf-8') as trace:
                    for line in trace:
                        step = json.loads(line)
                        for call in step.get('model_calls') or []:
                            if call['call_kind'] not in EXPERT_SITES:
                                raise ValueError(f'{key}: unexpected non-expert call in drive-mode capture')
                            row = site_row(call, step, key, output)
                            if row is None:
                                counts['ineligible_sites'] += 1
                                continue
                            if call['call_id'] in seen:
                                raise ValueError(f"{key}: duplicate call {call['call_id']}")
                            seen.add(call['call_id'])
                            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')
                counts['rows'] += len(seen)
                counts['episodes_with_rows'] += int(bool(seen))
        if not counts['rows']:
            raise ValueError('No eligible expert sites found in completed captures')
        report = dict(counts, terminals=len(terminals), excluded=excluded)
        (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Benchmark run directory containing episodes/')
    parser.add_argument('--output', required=True, type=Path, help='New directory for rows and content-addressed images')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(freeze(args.run, args.output), indent=2))
    except (KeyError, ValueError, OSError, TrainError) as exc:
        parser.exit(1, f'freeze: {exc}\n')


if __name__ == '__main__':
    main()
