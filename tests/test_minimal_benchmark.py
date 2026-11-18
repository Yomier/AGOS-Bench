"""Minimal-Agent runs through the same public scoring and resume paths as model arms."""
import json
from io import BytesIO

import pytest

from benchmark import paper_metrics, run_parallel_eval as evaluation
from benchmark.purpose import admission, manifest_purpose
from benchmark.scoring import read_terminal, sha256, write_json
from benchmark.tasks import ROOT


ARM = 'minimal_agent'
CONFIG = ROOT / 'agents/configs/minimal_agent.yaml'
EPISODE = 'lite_easy_00003'
PLAN_SHA = 'e2a8699d9c36440e92eee2ad3e7c150c4cd54198d1d24d03e95dd238b089cb65'


def test_minimal_cli_dry_run_selects_default_config(tmp_path, monkeypatch, capsys):
    # The extracted public task volume is external. Exercise the rest of the CLI,
    # including method construction, registry resolution and purpose admission.
    def load_tasks(root, keys, check_navigation):
        assert keys == [('test300', 'Town03/' + EPISODE)]
        return [dict(key='test300/Town03/' + EPISODE, plan_sha256=PLAN_SHA)], {}

    monkeypatch.setattr(evaluation, 'load_tasks', load_tasks)
    run_dir = tmp_path / 'run'
    assert evaluation.main(['--arm', ARM, '--dry-run', '--data-root', str(tmp_path), '--episode',
                            'Town03/' + EPISODE, '--run-dir', str(run_dir)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert (output['arm'], output['mode'], output['purpose']) == (ARM, ARM, 'evaluation')
    assert output['config'] == str(CONFIG) and not run_dir.exists()


def test_minimal_service_admission_uses_only_the_shared_model(monkeypatch):
    paths = []

    def models(request, **kwargs):
        paths.append(request.full_url)
        return BytesIO(b'{"data": [{"id": "qwen3.5-4b"}]}')

    monkeypatch.setattr(evaluation, 'urlopen', models)
    vlm = dict(api_url='http://127.0.0.1:8114/v1/chat/completions', model='qwen3.5-4b')
    config = {'vlm': vlm}
    evaluation.check_services(ARM, config, 'local')
    assert paths == ['http://127.0.0.1:8114/v1/models']
    vlm['model'] = 'missing'
    with pytest.raises(ValueError, match='missing'):
        evaluation.check_services(ARM, config, 'local')
    paths.clear()
    evaluation.check_services(ARM, config, 'api')
    assert paths == []
    vlm['api_url'] = ''
    with pytest.raises(ValueError, match='URL'):
        evaluation.check_services(ARM, config, 'api')


@pytest.mark.parametrize('arm,cooperative,collect', [
    ('baseline_a', {'arm': 'baseline_a'}, False), ('menu_oracle', {}, False),
    (ARM, {'arm': 'baseline_a'}, False), (ARM, {}, True),
    ('agos_b', {'arm': 'agos_b', 'expert': {'enabled': True}}, True),
])
def test_minimal_configuration_cannot_cross_method_boundaries(arm, cooperative, collect):
    config = {'paper_experiment': {'arm': ARM, 'agent_mode': ARM}}
    if cooperative:
        config['cooperative_search'] = cooperative
    with pytest.raises(ValueError):
        admission(config, arm, 'train1100', collect)


@pytest.fixture
def receipt(tmp_path):
    # Scoring fields from the selected Test300 Whiteboard success, raw SHA-256
    # 18e11e26b86f60918e11a7156138feae7ac0a737e26b762124b2505f243c22a5.
    # The method identity and release envelope below use the public run format.
    from agos.eval.metric_contract import metric_protocol
    task = dict(key='test300/Town03/' + EPISODE, split='test300', town='Town03', episode=EPISODE,
                plan_sha256=PLAN_SHA)
    manifest = dict(schema=2, arm=ARM, mode=ARM, purpose='evaluation', tasks=[task],
                    config_sha256=sha256(CONFIG), resolved_config_sha256='resolved')
    raw = dict(episode_id=EPISODE, mode='joint_minimal_agent',
               run_meta={'agent_mode': ARM, 'source_config_sha256': manifest['config_sha256']},
               paper_experiment={'arm': ARM, 'agent_mode': ARM}, evaluation_binding={'plan_sha256': PLAN_SHA},
               success=True, completed=True, confirm_count=1, failure_reason='SUCCESS', failure_stage=None,
               outcome_class='success', steps=4, vlm_calls=4, uav_vlm_calls=1, ugv_vlm_calls=3,
               vlm_http_attempts=4, uav_vlm_http_attempts=1, ugv_vlm_http_attempts=3,
               paper_fault_audit={'complete': True, 'events': [], 'failures': []},
               success_protocol={'id': 'AGOS_CONFIRM30_V1', 'require_los': True, 'max_confirm_distance_m': 30.,
                                 'require_gt_visible_in_stop_6view': True, 'require_stop_observe': False},
               ne_valid=True, ne_distance_m=11.67261960551522, metric_protocol=metric_protocol(),
               release_task_key=task['key'], release_plan_sha256=PLAN_SHA)
    for key in ('purpose', 'arm', 'config_sha256', 'resolved_config_sha256'):
        raw['release_' + key] = manifest[key]
    folder = tmp_path / 'episodes' / task['key']
    attempt = folder / 'attempt-0001'
    attempt.mkdir(parents=True)
    write_json(tmp_path / 'manifest.json', manifest)
    write_json(attempt / 'job.json', dict(manifest, task=task))
    write_json(attempt / (EPISODE + '_result.json'), raw)
    return manifest, task, folder, attempt, raw


def test_minimal_receipt_enters_paper_scoring_and_resume(tmp_path, receipt):
    manifest, task, folder, attempt, _ = receipt
    row = evaluation.collect_attempt(folder, attempt, task, ARM, manifest)
    assert row['outcome'] == 'success'
    assert read_terminal(folder, task, manifest) == row
    report = paper_metrics.score_run(tmp_path)
    assert report['purpose'] == 'evaluation' and report['complete']
    assert {k: report['metrics'][k] for k in ('SR', 'NE', 'DS', 'DSR')} == {
        'SR': 100., 'NE': pytest.approx(11.67261960551522), 'DS': 4., 'DSR': 25.}
    # Re-reading a locked result also rechecks expert/foreign method traces.
    (attempt / 'ag_os_trace.jsonl').write_text(json.dumps({'variant_id': 'baseline_a'}) + '\n')
    with pytest.raises(ValueError, match='trace arm'):
        read_terminal(folder, task, manifest)


def test_episode_child_dispatches_minimal_and_binds_receipt(receipt, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from agos import episode_io
    from agos.control import carla_interface
    from agos.lite_core import load_config
    from agos.runtime.runner import JointLiteRunner
    from benchmark.episode import run

    manifest, task, folder, attempt, raw = receipt
    plan = attempt / 'plan.json'
    write_json(plan, {'episode_id': EPISODE})
    task.update(path=str(plan), plan_sha256=sha256(plan))
    siblings = attempt / 'siblings.json'
    write_json(siblings, [])
    digest = hashlib.sha256(json.dumps(load_config(CONFIG), sort_keys=True, default=str).encode()).hexdigest()
    manifest['resolved_config_sha256'] = digest
    job = dict(manifest, task=task, config=str(CONFIG), output=str(attempt), rpc=2000, airsim=41451,
               siblings=str(siblings), slot=0)
    job_path = attempt / 'job.json'
    write_json(job_path, job)
    monkeypatch.setattr(episode_io, 'load_and_validate_episode_plan', lambda path, **kw: json.loads(plan.read_bytes()))
    monkeypatch.setattr(carla_interface, 'CarlaInterface', lambda *args: SimpleNamespace(cleanup=lambda: None))
    # Substitute only the simulator episode. The real child revalidates the YAML,
    # constructs JointLiteRunner and binds the result to the per-attempt job.
    def run_fixture(self, iface, episode, output):
        assert self.agent_mode == ARM and self.vlm_config == self.config['vlm']
        return dict(raw, evaluation_binding={'plan_sha256': task['plan_sha256']})

    monkeypatch.setattr(JointLiteRunner, 'run_episode', run_fixture)
    for key in ('AGOS_SOURCE_CONFIG_SHA256', 'AGOS_SOURCE_CONFIG_BASENAME', 'AGOS_EPISODE_PLAN_SHA256',
                'AGOS_EPISODE_MAP_KEY', 'AGOS_EPISODE_ID'):
        monkeypatch.setenv(key, '')
    run(job_path)
    assert evaluation.collect_attempt(folder, attempt, task, ARM, manifest)['outcome'] == 'success'
    write_json(job_path, dict(job, mode='cooperative_search'))
    with pytest.raises(ValueError, match='job identity'):
        run(job_path)


@pytest.mark.parametrize('changes', [
    {'mode': 'joint_cooperative_search'}, {'release_purpose': 'expert_collect'},
    {'paper_experiment': {'agent_mode': 'cooperative_search', 'arm': 'agos_b'}},
    {'paper_experiment': ['minimal_agent']},
    {'ugv_vlm_calls': 0}, {'vlm_http_attempts': 0}, {'oracle_candidate_injected': True},
    {'paper_fault_audit': {'complete': True, 'events': [{'kind': 'model_transport'}]}},
])
def test_inconsistent_minimal_receipts_remain_pending(receipt, changes):
    manifest, task, folder, attempt, raw = receipt
    write_json(attempt / (EPISODE + '_result.json'), dict(raw, **changes))
    assert evaluation.collect_attempt(folder, attempt, task, ARM, manifest)['outcome'] == 'pending'


@pytest.mark.parametrize('arm,mode,purpose', [(ARM, 'cooperative_search', 'evaluation'),
    ('baseline_a', ARM, 'evaluation'), (ARM, ARM, 'expert_collect'), (ARM, ARM, 'oracle_diagnostic')])
def test_minimal_manifest_rejects_relabelling(arm, mode, purpose):
    with pytest.raises(ValueError):
        manifest_purpose(dict(arm=arm, mode=mode, purpose=purpose, tasks=[]))
