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


def vision_counts(path, *, purpose=None, arm=None):
    errors = ok = 0
    if path.is_file():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                step = json.loads(line)
                if arm and step.get('variant_id', arm) != arm:
                    raise ValueError('method trace arm disagrees with the job')
                if purpose == 'evaluation' and ('expert_decision' in step or 'expert_label' in step or any(
                        call.get('call_kind', '').startswith('expert_') for call in step.get('model_calls') or [])):
                    raise ValueError('privileged expert trace in model evaluation')
                status = (step.get('vision') or {}).get('status')
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


def aggregate(outcomes, purpose='evaluation'):
    successes = sum(v == 'success' for v in outcomes.values())
    failures = sum(v == 'failure' for v in outcomes.values())
    pending = sorted(k for k, v in outcomes.items() if v == 'pending')
    complete = successes + failures
    rate = successes / complete if complete else None
    summary = dict(purpose=purpose, total=len(outcomes), successes=successes, failures=failures, pending=pending,
                   sr=rate if purpose == 'evaluation' else None, coverage=complete / len(outcomes) if outcomes else 0)
    if purpose == 'oracle_diagnostic':
        summary['oracle_qa_sr'] = rate
    return summary


def next_attempt(folder):
    numbers = [int(p.name[8:]) for p in folder.glob('attempt-*') if p.name[8:].isdigit()]
    return folder / f'attempt-{max(numbers, default=0) + 1:04d}'


def adjudicate(attempt, result, task, manifest, *, job=None):
    from .purpose import manifest_purpose, verify_calls
    purpose = manifest_purpose(manifest)
    if job is None:
        job = json.loads((attempt / 'job.json').read_text(encoding='utf-8'))
    for key in ('purpose', 'arm', 'config_sha256', 'resolved_config_sha256'):
        if job.get(key) != manifest[key] or result.get('release_' + key) != manifest[key]:
            raise ValueError(f'job or result {key} disagrees with manifest')
    mode = manifest['mode']
    plan = task['plan_sha256']
    key = task['key']
    dispatched = job['task']
    if (job.get('mode') != mode or dispatched['key'] != key or dispatched['plan_sha256'] != plan
            or key.split('/') != [dispatched.get(field) for field in ('split', 'town', 'episode')]
            or result.get('release_task_key') != key or result.get('release_plan_sha256') != plan):
        raise ValueError('inconsistent task, plan or mode binding')
    native = result.get('run_meta') or {}
    if (result.get('episode_id') != key.rsplit('/', 1)[-1] or native.get('agent_mode') != mode
            or result.get('mode') != 'joint_' + mode or native.get('source_config_sha256') != manifest['config_sha256']
            or (result.get('evaluation_binding') or {}).get('plan_sha256') != plan):
        raise ValueError('native runner result disagrees with task, plan, method or source config')
    errors, ok = vision_counts(attempt / 'ag_os_trace.jsonl', purpose=purpose, arm=manifest['arm'])
    outcome = classify(result, errors, ok)
    if outcome != 'pending':
        verify_calls(result, purpose)
    return outcome, errors, ok


def load_terminal(folder, task=None, manifest=None):
    path = folder / 'terminal.json'
    if not path.is_file():
        return None, None
    row = json.loads(path.read_text(encoding='utf-8'))
    raw = None
    if row['outcome'] not in {'success', 'failure', 'pending'}:
        raise ValueError('invalid terminal outcome')
    if row['outcome'] in {'success', 'failure'}:
        result = folder / row['attempt'] / row['result']
        if not result.resolve().is_relative_to(folder.resolve()):
            raise ValueError('locked result outside episode directory')
        data = result.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['result_sha256']:
            raise ValueError(f'locked result changed: {result}')
        raw = json.loads(data)
        if manifest is not None:
            if row.get('purpose') != manifest['purpose'] or result.name != task['key'].rsplit('/', 1)[-1] + '_result.json':
                raise ValueError('locked receipt disagrees with run purpose or episode')
            proof = adjudicate(result.parent, raw, task, manifest)
            if proof != (row['outcome'], row['vision_errors'], row['vision_ok']):
                raise ValueError('locked outcome disagrees with original evidence')
    return row, raw


def read_terminal(folder, task=None, manifest=None):
    return load_terminal(folder, task, manifest)[0]
