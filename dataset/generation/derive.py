"""Mint a custom ground-start variant of a released task in two fresh CarlaAir engines."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
from threading import Event

from agos.roadnet import RoadGraphStore
from agos.roadnet.registry import RoadGraphRegistry
from agos.scripts.semantic_nav_adapter import SemanticNavAdapter
from benchmark.scoring import sha256, write_json
from benchmark.tasks import ROOT, SPLITS, load_tasks, select_keys
from carlaair.engine.slot import BINARY, Slot, own_slot, run_owned


# These source fields describe the frozen scene and reference photographs. The
# original UGV route, deployment certificates and dataset identity are excluded.
SCENE_FIELDS = ('version', 'schema_version', 'geometry_contract', 'action_space_version',
                'control_graph', 'control_graph_release_id', 'control_graph_sha256',
                'sparse_intersection_max_distance_m', 'difficulty', 'vehicle_deployment_protocol',
                'observation_contract', 'map', 'region', 'region_policy_revision', 'target_task',
                'target_oracle', 'ugv_blueprint_pref', 'distractors_oracle', 'start',
                'uav_grid_waypoints', 'ugv_road_waypoints', 'road_graph_id', 'road_graph_map_key',
                'road_graph_asset_sha256', 'success_oracle', 'success_protocol', 'solvability_oracle',
                'ugv_identity', 'teammate_reference', 'target_reference')


def choose_start(parent, nodes, adapter, seed):
    """Sample a distinct in-region landing from the parent's executed ground route."""
    region = parent['region']
    target = parent['target_oracle']['transform']
    old = parent['start']['ugv']
    min_distance = 100.0 if parent.get('difficulty') in ('mid', 'hard') else 0.0
    ports = sorted({step['to_port'] for step in parent['navigation_oracle']['steps']})
    eligible = []
    for port in ports:
        node = nodes.get(port)
        if node is None or port not in adapter.port_owner or port == parent['ugv_start_port_fine_node']:
            continue
        if not (region['x_min'] <= node.x <= region['x_max'] and region['y_min'] <= node.y <= region['y_max']):
            continue
        if math.hypot(node.x - old['x'], node.y - old['y']) < 1:
            continue
        distance = math.hypot(node.x - target['x'], node.y - target['y'])
        if distance < min_distance or distance <= parent['success_oracle']['max_confirm_distance_m']:
            continue
        if any(math.hypot(node.x - d['x'], node.y - d['y']) <= 10 for d in parent['distractors_oracle']):
            continue
        eligible.append(port)
    if not eligible:
        raise ValueError('parent has no alternative in-region route landing outside the confirmation radius')
    port = random.Random(seed).choice(eligible)
    node = nodes[port]
    return port, adapter.port_owner[port], dict(x=node.x, y=node.y, z=node.z, yaw=node.yaw_deg)


def graphs(plan):
    town = plan['road_graph_map_key']
    assets = Path(os.environ.get('AGOS_ASSET_ROOT', ROOT / 'assets'))
    registry = RoadGraphRegistry.load(str(assets / 'navigation/road_graphs_v2/road_graph_registry.json'))
    directory = registry.resolve(town, plan['road_graph_id'], plan['road_graph_asset_sha256'])
    store = RoadGraphStore.load(directory, expected_map_key=town, expected_graph_id=plan['road_graph_id'])
    control = assets / 'navigation/ugv_control_graph' / town / 'ugv_control_graph.json'
    if store.manifest.asset_sha256 != plan['road_graph_asset_sha256'] or sha256(control) != plan['control_graph_sha256']:
        raise ValueError('frozen navigation graph hash mismatch')
    return store, SemanticNavAdapter.load(str(control))


def prepare(data_root, split, episode, seed):
    if not 0 <= seed < 2**63:
        raise ValueError('seed must be in [0, 2**63)')
    tasks, _ = load_tasks(Path(data_root).resolve(), select_keys(split, [episode]))
    source = tasks[0]
    raw = Path(source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['plan_sha256']:
        raise ValueError('released parent plan changed after verification')
    parent = json.loads(raw)
    store, adapter = graphs(parent)
    port, owner, pose = choose_start(parent, store.nodes, adapter, seed)
    plan = {k: parent[k] for k in SCENE_FIELDS if k in parent}
    plan['start']['ugv'] = pose
    for actor in [plan['target_oracle'], *plan['distractors_oracle']]:
        actor.pop('actor_id', None)
    plan.update(episode_id=f"derived_{source['plan_sha256'][:16]}_{seed}",
                dataset_id='AGOS_CUSTOM_GROUND_START_V1', ugv_start_control_node_id=owner,
                ugv_start_port_fine_node=port, navigation_oracle_required=True,
                semantic_adapter_sha256=sha256(ROOT / 'agos/scripts/semantic_nav_adapter.py'),
                generation_provenance=dict(kind='derived_ground_start_v1', seed=seed,
                    parent_split=split, parent_episode=episode, parent_plan_sha256=source['plan_sha256'],
                    inherited_reference_images=True))
    return plan, source, store, adapter


def run(data_root, split, episode, seed, output, sim_root, slot_id, dry_run):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'run directory already exists: {output}')
    if not 0 <= slot_id < 100:
        raise ValueError('choose a slot in [0, 99]')
    plan, source, _, _ = prepare(data_root, split, episode, seed)
    proposal = dict(episode_id=plan['episode_id'], parent_plan_sha256=source['plan_sha256'],
                    ugv_start_port=plan['ugv_start_port_fine_node'], ugv_start=plan['start']['ugv'])
    if dry_run:
        return dict(stage='offline_proposal', **proposal)
    if not sim_root:
        raise ValueError('live generation requires --sim-root or AGOS_SIM_ROOT')
    sim_root = Path(sim_root).resolve()
    engine_sha = sha256(sim_root / BINARY)
    if os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE'):
        from benchmark.native_runtime import load_profile
        profile = load_profile(os.environ['AGOS_NATIVE_RUNTIME_PROFILE'])
        if profile['engine']['sha256'] != engine_sha:
            raise ValueError('selected simulator differs from the native profile')
    slot = Slot(slot_id)
    stop = Event()
    with own_slot(slot):
        output.mkdir(parents=True, exist_ok=False)
        stage = output / 'candidate'
        stage.mkdir()
        src = Path(source['path']).parent
        for role in ('target_reference', 'teammate_reference'):
            ref = plan[role]
            rel = Path(ref['png_relpath'])
            if len(rel.parts) != 2 or rel.parts[0] != 'images' or rel.suffix != '.png':
                raise ValueError('unexpected released reference image path')
            dest = stage / rel
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(src / rel, dest)
            if sha256(dest) != ref['png_sha256']:
                raise ValueError(f'{role} changed during staging')
        draft = output / 'proposal.json'
        write_json(draft, plan)
        path = stage / (plan['episode_id'] + '_plan.json')
        digest = sha256(draft)
        prior = signal.signal(signal.SIGTERM, lambda *_: stop.set())
        try:
            for phase in ('generate', 'certify'):
                work = output / phase
                work.mkdir()
                input_path = draft if phase == 'generate' else path
                cmd = [sys.executable, '-m', 'dataset.generation.worker', phase, str(input_path),
                       str(stage), str(work), str(slot_id), digest]
                run_owned(slot, work, sim_root, source['town'], cmd, work, 7200, stop, lambda: None)
                if phase == 'generate':
                    digest = sha256(path)
        finally:
            signal.signal(signal.SIGTERM, prior)
        receipt = json.loads((output / 'certify/deployment_validation_receipt.json').read_text(encoding='utf-8'))
        if receipt.get('passed') is not True or receipt.get('actual_oracle_recomputed') is not True or sha256(path) != digest:
            raise ValueError('independent replay did not verify the generated plan')
        if receipt['frozen_certificate_sha256'] != json.loads(path.read_text(encoding='utf-8'))['navigation_oracle']['certificate_sha256']:
            raise ValueError('replay receipt identifies a different navigation certificate')
        accepted = dict(stage='independently_replayed_custom_episode', plan_sha256=digest,
                        engine_sha256=engine_sha, plan='episode/' + path.name, **proposal)
        stage.rename(output / 'episode')
        write_json(output / 'accepted.json', accepted)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--episode', required=True, metavar='TOWN/EPISODE')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new run directory')
    parser.add_argument('--sim-root', type=Path, default=os.getenv('AGOS_SIM_ROOT'))
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='validate parent and select a candidate start offline')
    args = parser.parse_args()
    try:
        result = run(args.data_root, args.split, args.episode, args.seed, args.output, args.sim_root, args.slot, args.dry_run)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
