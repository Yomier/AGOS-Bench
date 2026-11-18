import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.run_parallel_eval import parser, worker_count
from benchmark.tasks import select_keys
from benchmark.scoring import classify, aggregate, next_attempt
from carlaair.engine.slot import Slot, prepare_engine


def test_public_split_membership():
    rows = select_keys('all', [])
    assert len(rows) == len(set(rows)) == 2125
    assert ('test300', 'Town03/lite_easy_00001') in rows
    assert ('train1100', 'Town01/lite_easy_00001') in rows
    assert ('validation275', 'Town01/lite_easy_00001') in rows
    with pytest.raises(ValueError):
        select_keys('test300', ['Town01/lite_easy_00001'])


def test_serving_profile_defaults():
    defaults = parser().parse_args(['--run-dir', 'run'])
    assert worker_count(defaults.profile, defaults.workers) == 6
    assert worker_count('api', None) == 10
    assert worker_count('local', 1) == 1


def test_score_real_failures_and_exclude_infrastructure():
    def event(success, reason, stage, **extra):
        return dict(success=success, failure_reason=reason, failure_stage=stage, **extra)
    ok = event(True, 'SUCCESS', None, confirm_count=1, completed=True)
    assert classify(ok, 0, 1) == 'success'
    assert classify(ok, 2, 1) == 'pending'
    assert classify(dict(ok, outcome_class='infra_error')) == 'pending'
    assert classify(event(False, 'BUDGET_EXHAUSTED', 'STEP_BUDGET_EXHAUSTED', outcome_class='infra_error'), 0, 0) == 'pending'
    assert classify(dict(ok, confirm_count=0), 0, 0) == 'pending'
    assert classify(event(False, 'GUARD_STOP', 'VLM_BUDGET_EXHAUSTED'), 0, 0) == 'failure'
    for reason in ('BUDGET_EXHAUSTED', 'UGV_NAV_STALLED', 'UGV_PATROL_LOOP_ABORTED'):
        assert classify(event(False, reason, 'STEP_BUDGET_EXHAUSTED'), 1, 1) == 'failure'
    assert classify(event(False, 'UGV_CONFIRMED_WRONG_OBJECT', 'FALSE_CONFIRM'), 0, 0) == 'failure'
    for stage in ('EPISODE_WALL_TIMEOUT', 'SIM_TIME_EXHAUSTED', 'RUNTIME_EXCEPTION', 'NEW_STAGE'):
        assert classify(event(False, 'BUDGET_EXHAUSTED', stage), 0, 0) == 'pending'
    summary = aggregate({'a': 'success', 'b': 'failure', 'c': 'pending'})
    assert summary['sr'] == .5 and summary['coverage'] == 2/3 and summary['pending'] == ['c']


def test_faults_before_terminal_and_visual_error_ratio():
    from benchmark.scoring import vision_counts
    ok = dict(success=True, failure_stage=None, failure_reason='SUCCESS', confirm_count=1, completed=True)
    assert classify(dict(ok, runtime_error='RPC failed')) == 'pending'
    assert classify(dict(ok, paper_fault_audit={'complete': False, 'events': []})) == 'pending'
    assert classify(dict(ok, paper_fault_audit={'complete': True, 'events': [{'kind': 'model_transport'}]})) == 'pending'
    # A cleanup fault happened after the adjudicated CONFIRM.
    post = dict(ok, paper_fault_audit={'complete': True, 'events': [{'status': 'post_terminal_failure'}]})
    assert classify(post) == 'success'


def test_capture_root_and_dataset_id_are_scoped_to_attempt():
    from benchmark.episode import bind_episode
    config = {'benchmark': {}, 'paper_experiment': {}, 'cooperative_search': {'capture_root': 'captures'}}
    task = {'key': 'train1100/Town01/lite_easy_00001', 'split': 'train1100'}
    bind_episode(config, {'dataset_id': 'original-train-tree'}, task, Path('runs/attempt-0002'))
    assert config['benchmark']['dataset_id'] == config['paper_experiment']['dataset_id'] == 'original-train-tree'
    assert Path(config['cooperative_search']['capture_root']).as_posix().endswith('train1100/Town01/lite_easy_00001/attempt-0002')


def test_attempt_directories_are_preserved(tmp_path):
    a = next_attempt(tmp_path)
    assert a.name == 'attempt-0001'
    a.mkdir()
    (a / 'evidence').write_bytes(b'first')
    b = next_attempt(tmp_path)
    assert b.name == 'attempt-0002' and (a / 'evidence').read_bytes() == b'first'


def test_engine_contract_isolated_per_slot(tmp_path):
    slots = [Slot(i) for i in range(10)]
    assert len({p for s in slots for p in s.ports}) == 50
    for slot in slots[:2]:
        work = tmp_path / str(slot.index)
        cmd, env = prepare_engine(slot, work, Path('/sim'), 'Town03')
        assert 'Town03_Opt' in cmd and '-quality-level=Epic' in cmd
        assert 'LD_PRELOAD' not in env
        settings = Path(env['HOME']) / 'Documents/AirSim/settings.json'
        assert json.loads(settings.read_text())['ApiServerPort'] == slot.airsim
        assert settings.is_relative_to(work)


def test_nonzero_oracle_http_or_foreign_arm_remain_pending(tmp_path):
    from benchmark.run_parallel_eval import collect_attempt
    from benchmark.scoring import read_terminal
    folder = tmp_path / 'task'
    attempt = folder / 'attempt-0001'
    attempt.mkdir(parents=True)
    task = dict(key='test300/Town03/lite_easy_00001', episode='lite_easy_00001', plan_sha256='plan')
    manifest = dict(arm='menu_oracle', mode='menu_oracle', purpose='oracle_diagnostic', config_sha256='yaml',
                    resolved_config_sha256='resolved', tasks=[task])
    (attempt / 'job.json').write_text(json.dumps(dict(manifest, task=task)))
    result = dict(success=True, completed=True, confirm_count=1, failure_stage=None, failure_reason='SUCCESS',
                  release_task_key=task['key'], release_plan_sha256='plan', release_arm='menu_oracle',
                  release_purpose='oracle_diagnostic', episode_id=task['episode'], mode='joint_menu_oracle',
                  run_meta={'agent_mode': 'menu_oracle', 'source_config_sha256': 'yaml'}, evaluation_binding={'plan_sha256': 'plan'},
                  release_config_sha256='yaml', release_resolved_config_sha256='resolved', vlm_calls=0, vlm_http_attempts=0,
                  uav_vlm_calls=0, ugv_vlm_calls=0, uav_vlm_http_attempts=0, ugv_vlm_http_attempts=0)
    file = attempt / 'lite_easy_00001_result.json'
    for changes in ({'vlm_http_attempts': 1}, {'release_arm': 'baseline_a'}, {'release_resolved_config_sha256': 'other'}):
        file.write_text(json.dumps(dict(result, **changes)))
        assert collect_attempt(folder, attempt, task, 'menu_oracle', manifest)['outcome'] == 'pending'
    file.write_text(json.dumps(result))
    assert collect_attempt(folder, attempt, task, 'menu_oracle', manifest)['outcome'] == 'success'
    assert read_terminal(folder, task, manifest)['attempt'] == attempt.name
    with pytest.raises(ValueError):
        read_terminal(folder, task, dict(manifest, arm='baseline_a'))
    file.write_text(json.dumps(dict(result, steps=5)))
    with pytest.raises(ValueError):
        read_terminal(folder, task, manifest)


def test_watchdog_stops_only_its_real_child():
    import threading
    import time
    from carlaair.engine.slot import stop_process, wait_child
    cmd = [sys.executable, '-B', '-c', 'import time; time.sleep(60)']
    with subprocess.Popen(cmd) as other, subprocess.Popen(cmd) as child:
        try:
            with pytest.raises(TimeoutError):
                wait_child(child, other, time.monotonic(), threading.Event())
            stop_process(child)
            assert child.poll() is not None and other.poll() is None
        finally:
            stop_process(child)
            stop_process(other)


def test_service_preflight_uses_real_http_and_skips_unused_models():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    from benchmark.run_parallel_eval import check_services
    paths = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            result = {'ok': True} if self.path == '/health' else {'data': [{'id': 'uav'}, {'id': 'ugv'}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        def log_message(self, *args):
            pass
    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        url = 'http://127.0.0.1:' + str(server.server_port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            coop = {'gsam_endpoint': url + '/v1/gsam', 'ugv_endpoint': url + '/v1/ugv_views'}
            config = {'cooperative_search': coop, 'vlm': {'model': 'base', 'api_url': url + '/v1/chat/completions',
                                                        'uav_model': 'uav', 'ugv_model': 'ugv'}}
            check_services('menu_oracle', {}, 'local')
            assert not paths
            check_services('cooperative_search', config, 'local')
            assert paths[-1] == '/v1/models'
            config['vlm']['ugv_model'] = 'missing'
            with pytest.raises(ValueError, match='missing'):
                check_services('cooperative_search', config, 'local')
            paths.clear()
            coop['expert'] = {'enabled': True, 'mode': 'drive'}
            check_services('cooperative_search', config, 'local')
            assert paths and set(paths) == {'/health'}
        finally:
            server.shutdown()


def test_foreign_listener_rejected():
    import socket
    from carlaair.engine.slot import ports_free
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        slot = type('Ports', (), {'ports': [server.getsockname()[1]]})()
        assert not ports_free(slot)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux abstract socket owner lock')
def test_two_runs_cannot_own_one_slot():
    from carlaair.engine.slot import own_slot
    with own_slot(Slot(9)):
        with pytest.raises(OSError):
            with own_slot(Slot(9)):
                pytest.fail('second owner admitted')


@pytest.mark.parametrize('module', ['benchmark.run_parallel_eval', 'carlaair.serving.vllm', 'carlaair.perception.gsam', 'carlaair.engine.profile'])
def test_entrypoints_help_without_simulator(module):
    out = subprocess.run([sys.executable, '-B', '-m', module, '--help'], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert 'usage:' in out.stdout
