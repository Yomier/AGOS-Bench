"""Internal episode worker for the engine owned by dataset.generation.derive."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from agos.benchmark_protocol import validate_plan_success_protocol
from agos.control.carla_interface import CarlaInterface
from agos.episode.reference_photo import assert_actor_identity
from agos.episode_io import load_and_validate_episode_plan
from agos.eval.navigation_oracle import create_deployment_certificate, verify_deployed_certificate
from agos.lite_core import load_config
from benchmark.scoring import write_json
from benchmark.tasks import ROOT
from carlaair.engine.slot import Slot
from .derive import graphs


def require_owner(evidence, slot):
    state = json.loads((evidence / 'status.json').read_text(encoding='utf-8'))
    if (state.get('monitor_pid') != os.getppid() or state.get('slot') != slot.index
            or state.get('ports') != list(slot.ports) or state.get('attempt') != str(evidence)
            or state.get('stage') not in ('checking_camera_and_rpc', 'evaluating')):
        raise RuntimeError('generation worker must be launched by the current engine owner')


def work(phase, input_path, stage, evidence, slot, digest):
    require_owner(evidence, slot)
    raw = input_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('episode changed before engine admission')
    if phase == 'generate':
        plan = json.loads(raw)
    elif phase == 'certify':
        plan = load_and_validate_episode_plan(str(input_path), formal=True)
    else:
        raise ValueError(f'unknown generation phase {phase}')
    config = load_config(str(ROOT / 'agents/configs/agos_b.yaml'))
    config['benchmark']['dataset_id'] = plan['dataset_id']
    config['ugv']['sparse_intersection_max_distance_m'] = plan['sparse_intersection_max_distance_m']
    validate_plan_success_protocol(plan, config)
    store, adapter = graphs(plan)
    iface = CarlaInterface('127.0.0.1', slot.rpc, slot.airsim)
    actors = []
    try:
        if iface.world.get_map().name.split('/')[-1] != plan['map'].split('/')[-1]:
            raise RuntimeError('owned engine loaded a different map')
        target = iface.spawn_target_from_plan(plan)
        if target is None:
            raise RuntimeError('target deployment failed')
        actors.append(target)
        distractors = iface.spawn_distractors_from_plan(plan)
        actors.extend(distractors)
        if len(distractors) != len(plan['distractors_oracle']):
            raise RuntimeError('incomplete distractor deployment')
        iface.world.tick()
        iface.configure_ugv_identity(plan['ugv_identity'])
        pose = plan['start']['ugv']
        if iface.set_ugv_transform(pose['x'], pose['y'], pose['z'], pose['yaw'], strict_kinematic=True) is None:
            raise RuntimeError('new ground start could not be deployed')
        assert_actor_identity(iface._ugv_actor, plan['ugv_identity'])
        frame = iface.capture_ugv_front_image()
        if not frame:
            raise RuntimeError('new ground start produced no camera frame')
        (evidence / 'start_view.png').write_bytes(frame)
        args = (plan, config, store, adapter, iface, target, distractors, plan['control_graph_sha256'])
        if phase == 'certify':
            verify_deployed_certificate(*args, evidence_dir=evidence)
            return
        cert = create_deployment_certificate(*args)
        if not cert.get('production_replay', {}).get('final_confirmable'):
            raise ValueError('new start has no replayed confirmable route')
        plan['navigation_oracle'] = cert
        path = stage / (plan['episode_id'] + '_plan.json')
        write_json(path, plan)
        load_and_validate_episode_plan(str(path), formal=True)
    finally:
        try:
            for actor in actors:
                if actor.is_alive:
                    actor.destroy()
        finally:
            iface.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('generate', 'certify'))
    parser.add_argument('source', type=Path)
    parser.add_argument('stage', type=Path)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('slot', type=int)
    parser.add_argument('digest')
    args = parser.parse_args()
    work(args.phase, args.source, args.stage, args.evidence, Slot(args.slot), args.digest)
