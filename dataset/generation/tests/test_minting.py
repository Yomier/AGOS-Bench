"""Offline generation checks. Set AGOS_DATA_ROOT for the released-volume tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


KEY = 'Town03/lite_easy_00001'


def test_selects_different_in_region_nonconfirmable_port():
    from dataset.generation.derive import choose_start
    nodes = {k: SimpleNamespace(x=x, y=0, z=1, yaw_deg=30) for k, x in
             [('original', 45), ('too_close', 29), ('outside', 101), ('alternative', 70)]}
    plan = {'region': dict(x_min=-100, x_max=100, y_min=-100, y_max=100),
            'ugv_start_port_fine_node': 'original', 'start': {'ugv': {'x': 45, 'y': 0}},
            'target_oracle': {'transform': {'x': 0, 'y': 0}}, 'distractors_oracle': [],
            'success_oracle': {'max_confirm_distance_m': 30},
            'navigation_oracle': {'steps': [{'to_port': k} for k in nodes]}}
    adapter = SimpleNamespace(port_owner={p: 'owner' for p in nodes})
    port, owner, pose = choose_start(plan, nodes, adapter, 37)
    assert (port, owner, pose) == ('alternative', 'owner', dict(x=70, y=0, z=1, yaw=30))
    plan['difficulty'] = 'hard'
    with pytest.raises(ValueError, match='alternative'):
        choose_start(plan, nodes, adapter, 37)
    plan['difficulty'] = 'easy'
    del nodes['alternative']
    with pytest.raises(ValueError, match='alternative'):
        choose_start(plan, nodes, adapter, 37)


def test_unknown_episode_and_existing_destination_are_rejected(tmp_path):
    from dataset.generation.subset import assemble
    with pytest.raises(ValueError, match='unknown episode'):
        assemble(tmp_path, 'test300', ['../invented'], tmp_path / 'new', False)
    with pytest.raises(FileExistsError):
        assemble(tmp_path, 'test300', [KEY], tmp_path, False)


def test_worker_refuses_direct_attachment(tmp_path):
    from dataset.generation.worker import require_owner
    from carlaair.engine.slot import Slot
    evidence = tmp_path / 'generate'
    evidence.mkdir()
    state = dict(monitor_pid=-1, slot=0, attempt=str(evidence), ports=list(Slot(0).ports), stage='evaluating')
    status = evidence / 'status.json'
    status.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match='owner'):
        require_owner(evidence, Slot(0))
    state['monitor_pid'] = os.getppid()
    status.write_text(json.dumps(state))
    require_owner(evidence, Slot(0))


@pytest.fixture
def released_data():
    if not os.environ.get('AGOS_DATA_ROOT'):
        pytest.skip('requires the extracted Unseen-Test300 volume (AGOS_DATA_ROOT)')
    return Path(os.environ['AGOS_DATA_ROOT']).resolve()


def test_verified_subset_is_benchmark_readable_and_detects_image_damage(released_data, tmp_path):
    from benchmark.tasks import load_tasks
    from dataset.generation.subset import assemble
    dest = tmp_path / 'subset'
    assemble(released_data, 'test300', [KEY], dest, True)
    assert not dest.exists()
    subprocess.run([sys.executable, '-B', '-m', 'dataset.generation.subset', '--data-root', str(released_data),
                    '--split', 'test300', '--episode', KEY, '--output', str(dest)], check=True)
    env = dict(os.environ)
    env.pop('AGOS_SIM_ROOT', None)
    subprocess.run([sys.executable, '-B', '-m', 'benchmark.run_parallel_eval', '--data-root', str(dest),
                    '--split', 'test300', '--episode', KEY, '--arm', 'menu_oracle', '--workers', '1',
                    '--run-dir', str(tmp_path / 'smoke'), '--dry-run'], check=True, env=env)
    tasks, _ = load_tasks(dest, [('test300', KEY)])
    assert len(tasks) == 1
    assert (dest / 'tasks.csv').is_file()
    img = next(dest.rglob('*target_reference.png'))
    img.write_bytes(b'damaged')
    with pytest.raises(ValueError, match='differs'):
        assemble(dest, 'test300', [KEY], tmp_path / 'rejected', False)
    assert not (tmp_path / 'rejected').exists()


def test_real_proposal_has_new_geometry_and_no_inherited_navigation_receipt(released_data):
    from dataset.generation.derive import prepare
    plan, source, store, adapter = prepare(released_data, 'test300', KEY, 37)
    parent = json.loads(Path(source['path']).read_text(encoding='utf-8'))
    assert plan['episode_id'].startswith('derived_') and plan['dataset_id'] != parent['dataset_id']
    assert plan['start']['ugv'] != parent['start']['ugv']
    assert plan['start']['uav'] == parent['start']['uav']
    assert plan['ugv_start_control_node_id'] == adapter.port_owner[plan['ugv_start_port_fine_node']]
    assert store.nodes[plan['ugv_start_port_fine_node']].x == plan['start']['ugv']['x']
    assert all(k not in plan for k in ('navigation_oracle', 'generation_quality_certificate',
                                       'static_solvability_precheck', '_plan_dir'))
    assert plan['generation_provenance']['parent_plan_sha256'] == source['plan_sha256']
    assert plan['target_reference'] == parent['target_reference']
    assert plan['teammate_reference'] == parent['teammate_reference']


def test_unverified_run_has_no_published_episode(tmp_path, released_data, monkeypatch):
    from contextlib import nullcontext
    from dataset.generation import derive
    phases = []
    def fail_second(slot, work, sim_root, town, command, attempt, timeout, stop, before_start):
        phases.append(work.name)
        if work.name == 'certify':
            raise RuntimeError('independent replay failed')
        proposal = json.loads((work.parent / 'proposal.json').read_text(encoding='utf-8'))
        (work.parent / 'candidate' / (proposal['episode_id'] + '_plan.json')).write_text('{}')
    monkeypatch.setattr(derive, 'own_slot', lambda slot: nullcontext())
    monkeypatch.setattr(derive, 'run_owned', fail_second)
    engine = tmp_path / derive.BINARY
    engine.parent.mkdir(parents=True)
    engine.write_text('test-only lifecycle fixture')
    root = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='independent replay failed'):
        derive.run(released_data, 'test300', KEY, 37, root, tmp_path, 0, False)
    assert phases == ['generate', 'certify']
    assert not (root / 'episode').exists()
    assert not (root / 'accepted.json').exists()
