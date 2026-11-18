"""Release entry points must enforce the same authority and evidence at each boundary."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agos.eval.metric_contract import metric_protocol
from benchmark import paper_metrics, run_parallel_eval as evaluation
from benchmark.scoring import read_terminal, sha256, write_json
from benchmark.tasks import ROOT


def test_misspelled_key_fails_real_method_preflight(tmp_path):
    config = yaml.safe_load((ROOT / 'agents/configs/agos_b.yaml').read_text())
    config['cooperative_search']['rountine_progress'] = .5
    path = tmp_path / 'agent.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='rountine_progress'):
        evaluation.verify_config(path, 'agos_b', {'Town03'})


def test_standard_plan_rejects_maneuver_env(monkeypatch):
    from benchmark.tasks import navigation_preflight
    monkeypatch.setenv('UGV_ACTION_SPACE', 'MANEUVER_V2')
    _, config = evaluation.verify_config(ROOT / 'agents/configs/agos_b.yaml', 'agos_b', {'Town03'})
    with pytest.raises(RuntimeError, match='action-space contract mismatch'):
        navigation_preflight(config)({'action_space_version': 'sparse_nav_v1', 'map': 'Town03'})


@pytest.mark.parametrize('arm', ['agos_a', 'agos_b', 'agos_c', 'baseline_a'])
@pytest.mark.parametrize('expert_mode', ['drive', 'label'])
def test_privilege_requires_explicit_train_collection(arm, expert_mode):
    from benchmark.purpose import admission
    config = {'cooperative_search': {'arm': arm, 'expert': {'enabled': True, 'mode': expert_mode}}}
    with pytest.raises(ValueError, match='expert'):
        admission(config, arm, 'test300', False)
    with pytest.raises(ValueError, match='expert'):
        admission(config, arm, 'train1100', False)
    with pytest.raises(ValueError, match='Train1100'):
        admission(config, arm, 'test300', True)
    if arm == 'agos_b' and expert_mode == 'drive':
        assert admission(config, arm, 'train1100', True) == 'expert_collect'


def record(purpose='evaluation'):
    calls = 4 if purpose == 'evaluation' else 0
    arm = 'menu_oracle' if purpose == 'oracle_diagnostic' else 'agos_b'
    mode = 'menu_oracle' if arm == 'menu_oracle' else 'cooperative_search'
    split = 'train1100' if purpose == 'expert_collect' else 'test300'
    task = dict(key=f'{split}/Town03/example', split=split, town='Town03', episode='example', plan_sha256='plan')
    manifest = dict(schema=2, purpose=purpose, mode=mode, arm=arm, config_sha256='yaml',
                    resolved_config_sha256='resolved', tasks=[task])
    raw = dict(episode_id='example', mode='joint_' + mode, run_meta={'source_config_sha256': 'yaml', 'agent_mode': mode},
               evaluation_binding={'plan_sha256': 'plan'}, success=True, completed=True, confirm_count=1,
               failure_reason='SUCCESS', success_protocol={'id': 'AGOS_CONFIRM30_V1', 'max_confirm_distance_m': 30,
               'require_los': True, 'require_gt_visible_in_stop_6view': True, 'require_stop_observe': False},
               metric_protocol=metric_protocol(), vlm_calls=calls, uav_vlm_calls=calls, ugv_vlm_calls=0,
               vlm_http_attempts=calls, uav_vlm_http_attempts=calls, ugv_vlm_http_attempts=0,
               release_task_key=task['key'], release_plan_sha256='plan', release_purpose=purpose, release_arm=arm,
               release_config_sha256='yaml', release_resolved_config_sha256='resolved')
    return manifest, task, raw


def collect(root, manifest, task, raw):
    folder = root / 'episodes' / task['key']
    attempt = folder / 'attempt-0001'
    attempt.mkdir(parents=True, exist_ok=True)
    write_json(root / 'manifest.json', manifest)
    job = {key: manifest[key] for key in ('purpose', 'arm', 'mode', 'config_sha256', 'resolved_config_sha256')}
    write_json(attempt / 'job.json', dict(job, task=task))
    write_json(attempt / 'example_result.json', raw)
    return folder, evaluation.collect_attempt(folder, attempt, task, manifest['mode'], manifest)


@pytest.mark.parametrize('change', [dict(vlm_calls=0, uav_vlm_calls=0), dict(ugv_vlm_calls=5),
    dict(episode_id='other'), dict(run_meta={'source_config_sha256': 'other', 'agent_mode': 'cooperative_search'}),
    dict(mode='joint_menu_oracle'), dict(evaluation_binding={'plan_sha256': 'other'}), dict(release_purpose='expert_collect'),
    dict(oracle_candidate_injected=True)])
def test_collect_and_resume_check_native_evidence(tmp_path, change):
    manifest, task, raw = record()
    folder, valid = collect(tmp_path, manifest, task, raw)
    assert valid['outcome'] == 'success'
    _, invalid = collect(tmp_path, manifest, task, dict(raw, **change))
    assert invalid['outcome'] == 'pending'
    # A rewritten terminal and matching result checksum cannot establish a scientific outcome.
    result = folder / valid['attempt'] / valid['result']
    write_json(folder / 'terminal.json', dict(valid, result_sha256=sha256(result)))
    with pytest.raises(ValueError):
        read_terminal(folder, task, manifest)
    with pytest.raises(ValueError):
        paper_metrics.score_run(tmp_path)


def test_run_summary_counts_expert_success_without_model_sr():
    from benchmark.scoring import aggregate
    outcomes = {'a': 'success', 'b': 'failure'}
    expert = aggregate(outcomes, 'expert_collect')
    assert expert['successes'] == 1 and expert['sr'] is None and expert['purpose'] == 'expert_collect'
    oracle = aggregate(outcomes, 'oracle_diagnostic')
    assert oracle['sr'] is None and oracle['oracle_qa_sr'] == .5
    assert aggregate(outcomes, 'evaluation')['sr'] == .5


def test_zero_call_failure_and_oracle_qa_keep_distinct_purposes(tmp_path):
    manifest, task, raw = record()
    raw.update(success=False, failure_stage='STEP_BUDGET_EXHAUSTED', failure_reason='BUDGET_EXHAUSTED',
               vlm_calls=0, uav_vlm_calls=0, vlm_http_attempts=0, uav_vlm_http_attempts=0)
    _, row = collect(tmp_path, manifest, task, raw)
    assert row['outcome'] == 'failure' and paper_metrics.score_run(tmp_path)['metrics']['SR'] == 0
    manifest, task, raw = record('oracle_diagnostic')
    _, row = collect(tmp_path, manifest, task, raw)
    assert row['outcome'] == 'success'
    report = paper_metrics.score_run(tmp_path)
    assert report['purpose'] == 'oracle_diagnostic' and report['metrics']['DS'] is None


def test_paper_aggregation_refuses_collection_and_zero_call_model_success(tmp_path):
    manifest, task, raw = record('expert_collect')
    _, row = collect(tmp_path, manifest, task, raw)
    assert row['outcome'] == 'success'
    with pytest.raises(ValueError, match='expert_collect'):
        paper_metrics.score_run(tmp_path)
    with pytest.raises(ValueError):
        paper_metrics.aggregate_paper_metrics([raw], model_decisions=True)
    _, _, raw = record()
    raw.update(vlm_calls=0, uav_vlm_calls=0)
    with pytest.raises(ValueError):
        paper_metrics.aggregate_paper_metrics([raw], model_decisions=True)


@pytest.mark.parametrize('legacy', [False, True])
def test_fresh_drive_receipt_freezes_but_expert_trace_cannot_be_relabelled(tmp_path, legacy):
    from PIL import Image
    from agos.lite_core import LiteObservation
    from agos.methods.cooperative_search.policy import DecisionPolicy
    from training.freeze import freeze
    manifest, task, raw = record('expert_collect')
    folder, _ = collect(tmp_path, manifest, task, raw)
    attempt = folder / 'attempt-0001'
    image = tmp_path / 'frame.png'
    Image.new('RGB', (8, 8)).save(image)
    obs = LiteObservation('example', 1, 80, 'UAV')
    obs.target_reference_image = obs.agent_image = image.read_bytes()
    policy = DecisionPolicy({'vlm': {}, 'cooperative_search': {}}, capture_root=str(tmp_path / 'captures'))
    policy.record_expert_verify(obs, [], {'selected_proposal': 'NONE', 'target_verdict': 'UNKNOWN'}, 'verify')
    trace = dict(role='UAV', step=1, variant_id='agos_b', model_calls=policy.drain_events())
    (attempt / 'ag_os_trace.jsonl').write_text(json.dumps(trace) + '\n')
    assert evaluation.collect_attempt(folder, attempt, task, manifest['mode'], manifest)['outcome'] == 'success'
    if legacy:
        import hashlib
        from agos.lite_core import load_config
        yaml_file = tmp_path / 'expert.yaml'
        cfg = yaml.safe_load((ROOT / 'agents/configs/agos_b.yaml').read_text())
        cfg['cooperative_search']['expert'] = {'enabled': True, 'mode': 'drive'}
        yaml_file.write_text(yaml.safe_dump(cfg))
        source = sha256(yaml_file)
        resolved = hashlib.sha256(json.dumps(load_config(str(yaml_file)), sort_keys=True, default=str).encode()).hexdigest()
        manifest.update(schema=1, config_sha256=source, resolved_config_sha256=resolved)
        manifest.pop('purpose')
        raw.pop('release_purpose')
        raw.update(release_config_sha256=source, release_resolved_config_sha256=resolved)
        raw['run_meta']['source_config_sha256'] = source
        job = dict(manifest, config=str(yaml_file), task=task)
        write_json(attempt / 'job.json', job)
        result_file = attempt / 'example_result.json'
        write_json(result_file, raw)
        terminal = json.loads((folder / 'terminal.json').read_bytes())
        terminal.pop('purpose')
        write_json(folder / 'terminal.json', dict(terminal, result_sha256=sha256(result_file)))
        write_json(tmp_path / 'manifest.json', manifest)
        assert freeze(tmp_path, tmp_path / 'old-frozen')['rows'] == 1
        with pytest.raises(ValueError, match='purpose'):
            paper_metrics.score_run(tmp_path)
        yaml_file.write_text(yaml_file.read_text().replace('enabled: true', 'enabled: false'))
        with pytest.raises(ValueError, match='config'):
            freeze(tmp_path, tmp_path / 'wrong-config')
        return
    assert freeze(tmp_path, tmp_path / 'frozen')['rows'] == 1
    manifest['purpose'] = raw['release_purpose'] = 'evaluation'
    raw.update(vlm_calls=1, uav_vlm_calls=1, vlm_http_attempts=1, uav_vlm_http_attempts=1)
    assert collect(tmp_path, manifest, task, raw)[1]['outcome'] == 'pending'
    with pytest.raises(ValueError, match='expert_collect'):
        freeze(tmp_path, tmp_path / 'other')


def test_resume_rejects_job_or_trace_changes(tmp_path):
    manifest, task, raw = record()
    folder, locked = collect(tmp_path, manifest, task, raw)
    attempt = folder / locked['attempt']
    job_path = attempt / 'job.json'
    job = json.loads(job_path.read_bytes())
    write_json(job_path, dict(job, purpose='expert_collect'))
    with pytest.raises(ValueError, match='purpose'):
        read_terminal(folder, task, manifest)
    write_json(job_path, job)
    (attempt / 'ag_os_trace.jsonl').write_text(json.dumps({'vision': {'status': 'error'}}) + '\n')
    with pytest.raises(ValueError, match='outcome'):
        read_terminal(folder, task, manifest)


def launcher(*args):
    return subprocess.run([sys.executable, '-B', '-m', 'carlaair.serving.vllm', *map(str, args), '--dry-run'],
                          cwd=ROOT, capture_output=True, text=True)


def test_vllm_directory_and_split_capacity(tmp_path):
    assert launcher('--model-dir', '').returncode == 2
    base = tmp_path / 'Qwen3.5-4B'
    base.mkdir()
    assert launcher('--model-dir', base).returncode == 2
    write_json(base / 'config.json', {'model_type': 'qwen3_5'})
    assert launcher('--model-dir', base).returncode == 2
    index = base / 'model.safetensors.index.json'
    write_json(index, {'weight_map': {'a': 'model-00001.safetensors', 'b': 'model-00002.safetensors'}})
    (base / 'model-00001.safetensors').write_bytes(b'a')
    assert launcher('--model-dir', base).returncode == 2
    (base / 'model-00002.safetensors').write_bytes(b'b')
    single = launcher('--model-dir', base)
    assert single.returncode == 0 and '--enable-lora' not in single.stdout
    adapters = []
    for role in ('uav', 'ugv'):
        adapter = tmp_path / 'split' / role / 'adapter'
        adapter.mkdir(parents=True)
        write_json(adapter / 'adapter_config.json', {'peft_type': 'LORA', 'r': 8})
        adapters.extend(['--adapter', f'{role}={adapter}'])
        assert launcher('--model-dir', base, *adapters).returncode == 2
        (adapter / 'adapter_model.safetensors').write_bytes(b'a')
    split = launcher('--model-dir', base, *adapters)
    assert split.returncode == 0, split.stderr
    argv = shlex.split(split.stdout)
    assert argv[argv.index('--max-loras') + 1] == argv[argv.index('--max-cpu-loras') + 1] == '2'
    assert argv[argv.index('--lora-modules') + 1:] == adapters[1::2]
