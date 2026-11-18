"""Child process for one v13 runner invocation on an already-admitted slot."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .scoring import sha256, write_json


def bind_episode(config, plan, task, attempt):
    dataset_id = plan.get('dataset_id') or (config['benchmark'].get('dataset_id') if task['split'] == 'test300' else task['split'])
    config.setdefault('evaluation', {})['task_attempt'] = int(attempt.name.removeprefix('attempt-'))
    config['benchmark']['dataset_id'] = dataset_id
    config.setdefault('paper_experiment', {})['dataset_id'] = dataset_id
    coop = config.get('cooperative_search', {})
    if coop.get('capture_root'):
        coop['capture_root'] = str(Path(coop['capture_root']).resolve() / task['key'] / attempt.name)


def run(job_file):
    from agos.lite_core import load_config
    from agos.episode_io import load_and_validate_episode_plan
    from agos.control.carla_interface import CarlaInterface
    from agos.runtime.runner import JointLiteRunner
    job = json.loads(Path(job_file).read_text(encoding='utf-8'))
    task = job['task']
    if sha256(task['path']) != task['plan_sha256'] or sha256(job['config']) != job['config_sha256']:
        raise ValueError('episode plan or configuration changed after admission')
    config = load_config(job['config'])
    resolved_sha = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    if resolved_sha != job['resolved_config_sha256']:
        raise ValueError('resolved agent configuration changed after admission')
    config['carla'].update(host='127.0.0.1', rpc_port=job['rpc'], airsim_port=job['airsim'])
    os.environ.update(AGOS_SOURCE_CONFIG_SHA256=job['config_sha256'],
                      AGOS_SOURCE_CONFIG_BASENAME=Path(job['config']).name,
                      AGOS_EPISODE_PLAN_SHA256=task['plan_sha256'], AGOS_EPISODE_MAP_KEY=task['town'],
                      AGOS_EPISODE_ID=task['episode'])
    plan = load_and_validate_episode_plan(task['path'], formal=True, check_reference=True)
    bind_episode(config, plan, task, Path(job['output']))
    siblings = json.loads(Path(job['siblings']).read_text(encoding='utf-8'))
    siblings = [row for row in siblings if row['episode_id'] != task['episode']]
    started = time.monotonic()
    iface = CarlaInterface('127.0.0.1', job['rpc'], job['airsim'])
    try:
        runner = JointLiteRunner(config, agent_mode=job['mode'], sibling_episodes=siblings,
                                 vlm_config=config.get('vlm'))
        result = runner.run_episode(iface, plan, job['output'])
        result['episode_wall_time_s'] = time.monotonic() - started
        result['release_task_key'] = task['key']
        result['release_plan_sha256'] = task['plan_sha256']
        result['release_arm'] = job['arm']
        result['release_config_sha256'] = job['config_sha256']
        result['release_resolved_config_sha256'] = resolved_sha
        result['infra'] = dict(result.get('infra') or {}, slot_id=job['slot'], rpc_port=job['rpc'],
                               airsim_port=job['airsim'], map_key=task['town'])
        write_json(Path(job['output']) / (task['episode'] + '_result.json'), result)
    finally:
        iface.cleanup()


if __name__ == '__main__':
    run(sys.argv[1])
