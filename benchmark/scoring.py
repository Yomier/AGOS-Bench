"""Score one isolated attempt and aggregate only locked scientific outcomes."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def vision_counts(path):
    errors = ok = 0
    if path.is_file():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                status = (json.loads(line).get('vision') or {}).get('status')
                errors += status == 'error'
                ok += status == 'ok'
    return errors, ok


def classify(result, vision_errors=0, vision_ok=0):
    if not isinstance(result, dict) or vision_errors > vision_ok:
        return 'pending'
    if result.get('runtime_error'):
        return 'pending'
    audit = result.get('paper_fault_audit')
    if audit is not None:
        if audit.get('complete') is not True or audit.get('failures'):
            return 'pending'
        for event in audit.get('events', []):
            if event.get('status') == 'post_terminal_failure':
                continue
            # The published majority-error rule allows isolated transient vision errors.
            if (event.get('kind') == 'tool_service' and event.get('component', '').split('.')[0] == 'vision'
                    and event.get('status') == 'error' and vision_errors > 0):
                continue
            return 'pending'
    stage = result.get('failure_stage')
    reason = result.get('failure_reason')
    outcome = result.get('outcome_class')
    if result.get('success') is True:
        if outcome not in (None, 'success'):
            return 'pending'
        confirmed = result.get('confirm_count')
        if (stage in (None, '') and reason == 'SUCCESS' and result.get('completed') is True
                and type(confirmed) is int and confirmed > 0):
            return 'success'
        return 'pending'
    if result.get('success') is not False:
        return 'pending'
    # v13 serializes the model-call limit as a guard stop with completed=False.
    call_limit = stage == 'VLM_BUDGET_EXHAUSTED' and outcome in (None, 'guard_stop', 'real_fail')
    if not call_limit and (outcome not in (None, 'real_fail') or result.get('completed') is False):
        return 'pending'
    failures = {
        'FALSE_CONFIRM': {'UGV_CONFIRMED_WRONG_OBJECT'},
        'STEP_BUDGET_EXHAUSTED': {'BUDGET_EXHAUSTED', 'UGV_NAV_STALLED', 'UGV_PATROL_LOOP_ABORTED'},
        'VLM_BUDGET_EXHAUSTED': {'GUARD_STOP', 'BUDGET_EXHAUSTED', 'VLM_BUDGET_EXHAUSTED'},
        'UGV_NAV_STALLED': {'UGV_NAV_STALLED'},
        'UGV_PATROL_LOOP_ABORTED': {'UGV_PATROL_LOOP_ABORTED'},
    }
    return 'failure' if reason in failures.get(stage, set()) else 'pending'


def aggregate(outcomes):
    successes = sum(v == 'success' for v in outcomes.values())
    failures = sum(v == 'failure' for v in outcomes.values())
    pending = sorted(k for k, v in outcomes.items() if v == 'pending')
    complete = successes + failures
    return dict(total=len(outcomes), successes=successes, failures=failures, pending=pending,
                sr=successes / complete if complete else None,
                coverage=complete / len(outcomes) if outcomes else 0)


def next_attempt(folder):
    numbers = [int(p.name[8:]) for p in folder.glob('attempt-*') if p.name[8:].isdigit()]
    return folder / f'attempt-{max(numbers, default=0) + 1:04d}'


def verify_binding(result, task, manifest):
    expected = dict(release_task_key=task['key'], release_plan_sha256=task['plan_sha256'],
                    release_arm=manifest['arm'], release_config_sha256=manifest['config_sha256'],
                    release_resolved_config_sha256=manifest['resolved_config_sha256'])
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError('missing or inconsistent task, arm or configuration binding')


def read_terminal(folder, task=None, manifest=None):
    path = folder / 'terminal.json'
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding='utf-8'))
    if row['outcome'] in {'success', 'failure'}:
        result = folder / row['attempt'] / row['result']
        if sha256(result) != row['result_sha256']:
            raise ValueError(f'locked result changed: {result}')
        if manifest is not None:
            verify_binding(json.loads(result.read_text(encoding='utf-8')), task, manifest)
    return row
