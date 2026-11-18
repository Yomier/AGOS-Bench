"""Meter-optimal SPL oracle over the production semantic maneuver menu.

Evaluation-only code: actor geometry, LOS and certificates never enter model
observations. Generation's budget-optimized reachability route is independent.
Distances are horizontal meters along unrounded fine-node coordinates, exactly
as accumulated by the R33 executor. No region restriction is added here.
"""
from __future__ import annotations

import hashlib
import heapq
import inspect
import json
import math
import copy
import os
from agos.eval.deployment_validation import (LAYERED_PROTOCOL, LAYERED_ORACLE_VERSION,
    DEPLOYMENT_POLICY, compare_deployments, LAYERED_PROTOCOLS, LAYERED_ORACLE_VERSIONS,
    TASK_SCALE_PROTOCOL, TASK_SCALE_ORACLE_VERSION, deployment_policy, _decimal_delta)
from agos.benchmark_protocol import DEFAULT_CONFIRM_DISTANCE_M, confirm_distance_m
from pathlib import Path

ORACLE_VERSION = "semantic_meters_confirm_v1"
ZERO_TOLERANCE_M = 1e-6
CONSISTENCY_TOLERANCE_M = 1e-4


class NavigationOracleError(RuntimeError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def fine_path_distance(nodes, fine_path):
    """Shared executor/oracle metric, not menu text or rounded edge weights."""
    if not fine_path:
        raise NavigationOracleError("empty semantic fine path")
    distance = 0.0
    for a, b in zip(fine_path, fine_path[1:]):
        na, nb = nodes[a], nodes[b]
        distance += math.hypot(float(nb.x) - float(na.x), float(nb.y) - float(na.y))
    if not math.isfinite(distance):
        raise NavigationOracleError("nonfinite semantic path length")
    return distance


def _yaw(value):
    value = float(value)
    if not math.isfinite(value):
        raise NavigationOracleError("nonfinite navigation yaw")
    return round((value + 180.0) % 360.0 - 180.0, 6)


def shortest_confirmable_route(adapter, nodes, start_port, start_yaw,
                               goal_test, intersection_max_distance_m=50.0,
                               initial_goal_test=None):
    """Dijkstra over legal decision landing states (port, actual arrival yaw).

    Only popped states are tested. FORWARD_2 pass-through nodes are deliberately
    not candidate goals; REVERSE retains its production yaw override. Menu
    exceptions are fatal, because treating them as absent edges invalidates L*.
    """
    # Initial physical height can differ from later graph-landing height.
    # Keep it separate, including when a positive-length loop returns here.
    start = (start_port, _yaw(start_yaw), True)
    if start_port not in nodes:
        raise NavigationOracleError("SPL start not in frozen fine graph")
    distances = {start: 0.0}
    parents = {}
    counter = 0
    queue = [(0.0, counter, start)]
    checked = 0
    menus = {}
    while queue:
        distance, _, state = heapq.heappop(queue)
        if distance != distances.get(state):
            continue
        port, yaw, initial = state
        checked += 1
        test = initial_goal_test if initial and initial_goal_test is not None else goal_test
        if test(port, yaw):
            steps = []
            cur = state
            while cur != start:
                prev, step = parents[cur]
                steps.append(step)
                cur = prev
            steps.reverse()
            fine_path = [start_port]
            for step in steps:
                fine_path.extend(step['fine_path'][1:])
            return {'distance_m': distance, 'goal': {'port': port, 'yaw_deg': yaw},
                    'start': {'port': start_port, 'yaw_deg': start[1]},
                    'actions': [s['action'] for s in steps], 'steps': steps,
                    'fine_path': fine_path, 'settled_states': checked,
                    'goal_pose_kind': 'initial_spawn' if initial else 'semantic_landing'}
        if port not in menus:
            menus[port] = adapter.get_maneuver_options(port,
                intersection_max_distance_m=intersection_max_distance_m)
        for action, option in sorted(menus[port].items()):
            path = list(option.fine_path)
            target = option.target_fine_node_id
            if not path or path[0] != port or path[-1] != target or target not in nodes:
                raise NavigationOracleError("semantic menu fine_path violates landing contract")
            cost = fine_path_distance(nodes, path)
            arrival_yaw = _yaw(option.yaw_override_deg if option.yaw_override_deg is not None
                               else nodes[target].yaw_deg)
            next_state = (target, arrival_yaw, False)
            nd = distance + cost
            if nd < distances.get(next_state, math.inf):
                distances[next_state] = nd
                parents[next_state] = (state, {'action': action, 'fine_path': path,
                    'distance_m': cost, 'to_port': target, 'yaw_deg': arrival_yaw})
                counter += 1
                heapq.heappush(queue, (nd, counter, next_state))
    return None


def build_certificate(route, binding):
    if route is None:
        raise NavigationOracleError("NO_LEGAL_CONFIRM_STATE")
    cert = {'version': binding.get('oracle_version', ORACLE_VERSION), 'binding': copy.deepcopy(binding),
            'binding_sha256': canonical_hash(binding), **route}
    cert['certificate_sha256'] = canonical_hash(cert)
    return cert


def validate_certificate(certificate, binding):
    if not isinstance(certificate, dict) or certificate.get('version') not in (ORACLE_VERSION, *LAYERED_ORACLE_VERSIONS):
        raise NavigationOracleError("missing or unsupported navigation_oracle certificate")
    content = {k: v for k, v in certificate.items() if k != 'certificate_sha256'}
    try:
        intact = (certificate.get('certificate_sha256') == canonical_hash(content)
                  and certificate.get('binding_sha256') == canonical_hash(certificate.get('binding')))
    except (ValueError, TypeError) as error:
        raise NavigationOracleError('invalid navigation_oracle certificate content') from error
    if not intact:
        raise NavigationOracleError("navigation_oracle certificate content hash mismatch")
    layered = certificate['version'] in LAYERED_ORACLE_VERSIONS
    if layered:
        if (certificate['binding'].get('oracle_version') != certificate['version'] or
                binding.get('oracle_version') != certificate['version']):
            raise NavigationOracleError('changed oracle version')
        protocol = TASK_SCALE_PROTOCOL if certificate['version'] == TASK_SCALE_ORACLE_VERSION else LAYERED_PROTOCOL
        if certificate['binding'].get('deployment_validation') != deployment_policy(protocol):
            raise NavigationOracleError('changed oracle deployment policy')
        compare_deployments(certificate['binding'], binding)
    try:
        matches = layered or certificate.get('binding_sha256') == canonical_hash(binding)
    except (ValueError, TypeError) as error:
        raise NavigationOracleError('invalid navigation_oracle deployment binding') from error
    if not matches:
        def changed_paths(expected, actual, prefix=''):
            if isinstance(expected, dict) and isinstance(actual, dict):
                out = []
                for key in sorted(set(expected) | set(actual)):
                    path = prefix+'.'+str(key) if prefix else str(key)
                    if key not in expected or key not in actual:
                        out.append(path)
                    else:
                        out.extend(changed_paths(expected[key], actual[key], path))
                return out
            return [] if expected == actual else [prefix]
        paths = changed_paths(certificate.get('binding', {}), binding)
        raise NavigationOracleError("navigation_oracle deployment/config/code binding mismatch: "
                                    + ', '.join(paths[:12]))
    distance = certificate.get('distance_m')
    if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
        raise NavigationOracleError("invalid navigation_oracle meter optimum")
    return certificate


def spl_measurement(success, optimum_m, executed_m):
    """Return an explicit invalid metric instead of clipping contradictions."""
    if (not isinstance(optimum_m, (int, float)) or not isinstance(executed_m, (int, float))
            or not math.isfinite(optimum_m) or not math.isfinite(executed_m)
            or optimum_m < 0 or executed_m < 0):
        return {'value': None, 'valid': False, 'reason': 'invalid_distance'}
    if success and executed_m + CONSISTENCY_TOLERANCE_M < optimum_m:
        return {'value': None, 'valid': False, 'reason': 'executed_shorter_than_oracle'}
    if not success:
        value = 0.0
    elif optimum_m <= ZERO_TOLERANCE_M:
        value = 1.0 if executed_m <= ZERO_TOLERANCE_M else 0.0
    else:
        value = optimum_m / max(optimum_m, executed_m)
    return {'value': value, 'valid': True, 'reason': None}


def _actor_snapshot(actor):
    if actor is None or not getattr(actor, 'is_alive', False):
        raise NavigationOracleError("oracle requires all live authoritative deployment actors")
    tf, bb = actor.get_transform(), actor.bounding_box
    def vector(v):
        return [round(float(getattr(v, k)), 6) for k in ('x', 'y', 'z')]
    return {'blueprint': actor.type_id,
            'attributes': {k: v for k, v in sorted(dict(actor.attributes).items())
                           if k not in {'role_name', 'ros_name'}},
            'location': vector(tf.location),
            'rotation': [round(float(getattr(tf.rotation, k)), 6)
                         for k in ('pitch', 'yaw', 'roll')],
            'bbox_location': vector(bb.location), 'bbox_extent': vector(bb.extent)}


def _source_hash(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def replay_semantic_route(route, adapter, nodes, iface, intersection_max_distance_m=50):
    """Replay every authoritative option's fine path using the production setter.

    This is QA evidence, never an agent action. Caller owns restore/cleanup.
    Menu legality, landing yaw and exact meter accumulation are checked again.
    """
    port = route['start']['port']
    total = 0.0
    final_yaw = route['start']['yaw_deg']
    for step in route['steps']:
        option = adapter.get_maneuver_options(port,
            intersection_max_distance_m=intersection_max_distance_m).get(step['action'])
        if option is None or list(option.fine_path) != step['fine_path']:
            raise NavigationOracleError("certificate replay menu/path mismatch")
        path = list(option.fine_path)
        cost = fine_path_distance(nodes, path)
        for index, nid in enumerate(path[1:]):
            n = nodes[nid]
            yaw = option.yaw_override_deg if option.yaw_override_deg is not None else n.yaw_deg
            tf = iface.set_ugv_transform(n.x, n.y, n.z, yaw,
                fast_teleport=index != len(path) - 2, strict_kinematic=True)
            if tf is None:
                raise NavigationOracleError("certificate production path replay failed")
            if math.hypot(tf.location.x - n.x, tf.location.y - n.y) > 1e-4:
                raise NavigationOracleError("certificate replay deviated from legal fine node")
            final_yaw = _yaw(tf.rotation.yaw)
        total += cost
        port = option.target_fine_node_id
    if (abs(total - route['distance_m']) > CONSISTENCY_TOLERANCE_M
            or port != route['goal']['port']
            or abs(_yaw(final_yaw - route['goal']['yaw_deg'])) > 1e-3):
        raise NavigationOracleError("certificate replay distance/landing mismatch")
    return {'valid': True, 'executed_distance_m': total,
            'actions_replayed': len(route['steps']), 'final_port': port,
            'final_yaw_deg': final_yaw}


def oracle_required(plan, config):
    return (bool(plan.get('navigation_oracle_required', False))
            or (plan.get('teammate_reference') or {}).get('protocol') == 'AGOS_TEAMMATE_TOPDOWN_V1'
            or bool((config.get('evaluation', {}) or {}).get('require_navigation_oracle', False))
            or bool((config.get('benchmark', {}) or {}).get('navigation_oracle_required', False))
            or (config.get('observation_map', {}) or {}).get('variant_id') == 'R33_SHARED_MAP')


def deployment_binding(plan, config, store, adapter, iface, target_actor,
                       distractor_actors=(), control_graph_sha=''):
    from agos.eval.eval import SuccessChecker
    from agos.perception.perception import check_visibility_6view, check_visibility
    ugv = config.get('ugv', {}) or {}
    success = config.get('success', {}) or {}
    start_port = plan.get('ugv_start_port_fine_node')
    if not start_port or adapter is None:
        raise NavigationOracleError("new SPL requires production sparse_nav_v1 start/adapter")
    if len(distractor_actors) != len(plan.get('distractors_oracle', [])):
        raise NavigationOracleError("SPL deployment missing planned distractor actors")
    manifest = store.manifest
    nodes = [[str(nid), float(n.x), float(n.y), float(n.z), float(n.yaw_deg)]
             for nid, n in sorted(store.nodes.items())]
    protocol = (config.get('evaluation', {}) or {}).get('deployment_validation_protocol')
    if protocol not in (None, *LAYERED_PROTOCOLS):
        raise NavigationOracleError('unsupported deployment validation protocol')
    version = TASK_SCALE_ORACLE_VERSION if protocol == TASK_SCALE_PROTOCOL else LAYERED_ORACLE_VERSION
    result = {'oracle_version': version if protocol else ORACLE_VERSION,
        'oracle_source_sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'task': {'episode_id': plan.get('episode_id'), 'map': plan.get('map'),
                 'start': plan['start'], 'target': plan['target_oracle'],
                 'distractors': plan.get('distractors_oracle', [])},
        'world_map': iface.get_world().get_map().name,
        'start_port': start_port,
        'target_deployment': _actor_snapshot(target_actor),
        'distractor_deployment': [_actor_snapshot(a) for a in distractor_actors],
        'ugv_deployment': _actor_snapshot(iface._ugv_actor),
        'road_graph': {'id': manifest.graph_id, 'asset_sha': manifest.asset_sha256,
                       'fine_nodes_sha': canonical_hash(nodes)},
        'control_graph_sha': control_graph_sha or canonical_hash(adapter.graph),
        'semantic_adapter_sha': hashlib.sha256((Path(__file__).resolve().parents[1]
            / 'scripts' / 'semantic_nav_adapter.py').read_bytes()).hexdigest(),
        'intersection_max_distance_m': float(ugv.get('sparse_intersection_max_distance_m',
                                                     ugv.get('intersection_max_distance_m', 50))),
        'success': {k: success.get(k, default) for k, default in (
            ('require_stop_observe', False), ('require_gt_visible_in_stop_6view', True),
            ('require_los', False), ('max_confirm_distance_m', DEFAULT_CONFIRM_DISTANCE_M))},
        'camera': {'yaws': ugv.get('stop_observe_yaws', [-45, 0, 45, -135, 180, 135]),
                   'fov': ugv.get('stop_camera_fov_deg', 90),
                   'width': ugv.get('stop_panel_width', 480),
                   'height': ugv.get('stop_panel_height', 270),
                   'anchor': 'bbox_roof_plus_0.3m'},
        'los_fail_closed': bool((config.get('perception', {}) or {}).get('los_fail_closed', False)),
        'success_checker_sha': _source_hash(SuccessChecker),
        'sixview_sha': _source_hash(check_visibility_6view),
        'visibility_sha': _source_hash(check_visibility),
        'perception_source_sha': hashlib.sha256((Path(__file__).resolve().parents[1]
            / 'perception' / 'perception.py').read_bytes()).hexdigest(),
        'camera_anchor_source_sha': _source_hash(type(iface)._get_ugv_camera_anchor),
        'confirm_geometry_version': 'sixview_bbox_center_xy_roof_z_v1',
        'executor_sha': _source_hash(type(iface).set_ugv_transform),
        # Helpers outside the setter can change physical execution/spawn too.
        'control_module_source_sha': hashlib.sha256((Path(__file__).resolve().parents[1]
            / 'control' / 'carla_interface.py').read_bytes()).hexdigest(),
        'distance_metric': 'unrounded_fine_path_xy_m',
        'distance_source_sha': _source_hash(fine_path_distance),
        'zero_tolerance_m': ZERO_TOLERANCE_M,
        'consistency_tolerance_m': CONSISTENCY_TOLERANCE_M,
        'legal_region_policy': 'production_menu_no_task_region_clip'}
    if protocol:
        result['deployment_validation'] = deployment_policy(protocol)
        result['deployment_validation_source_sha'] = hashlib.sha256(
            Path(__file__).with_name('deployment_validation.py').read_bytes()).hexdigest()
        result['native_runtime_profile_sha256'] = os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE_SHA256')
    return result


def create_deployment_certificate(plan, config, store, adapter, iface, target_actor,
                                  distractor_actors=(), control_graph_sha=''):
    """Explicit QA/generation preflight after production actor deployment.

    Restores the existing kinematic UGV in finally, including on LOS errors.
    This function does not mutate plan or enable runtime fallback generation.
    """
    from agos.eval.eval import SuccessChecker
    from agos.perception.perception import check_visibility_6view
    binding = deployment_binding(plan, config, store, adapter, iface, target_actor,
                                  distractor_actors, control_graph_sha)
    checker = SuccessChecker(config)
    world = iface.get_world()
    actor = iface._ugv_actor
    original = actor.get_transform()
    def restore_start():
        if binding['oracle_version'] in LAYERED_ORACLE_VERSIONS:
            # Readback Euler values are observations, not immutable commands.
            # Rewriting them can accumulate quaternion conversion error and
            # change the initial scene as a side effect of private Oracle QA.
            command = plan['start']['ugv']
            restored = iface.set_ugv_transform(command['x'], command['y'],
                command['z'], command['yaw'], fast_teleport=True, strict_kinematic=True)
            if restored is None:
                raise NavigationOracleError('oracle immutable start restoration failed')
        else:
            actor.set_transform(original)
        world.tick()
    target_xy = target_actor.get_location()
    camera = binding['camera']
    goals = {}
    tested_goals = []
    def pose_goal(port, yaw, initial=False):
        n = store.nodes[port]
        x, y, z = ((original.location.x, original.location.y, original.location.z)
                   if initial else (n.x, n.y, n.z))
        distance = math.hypot(x - target_xy.x, y - target_xy.y)
        if distance > checker.max_confirm_dist:
            tested_goals.append([port, None if initial else yaw, initial, False])
            return False
        if not checker.require_gt_visible and not checker.require_los:
            tested_goals.append([port, None if initial else yaw, initial, True])
            return True
        if not initial:
            tf = iface.set_ugv_transform(x, y, z, yaw,
                                        fast_teleport=True, strict_kinematic=True)
            if tf is None:
                raise NavigationOracleError("oracle UGV production transform failed")
        visible, views, debug, los = check_visibility_6view(target_actor,
            (x, y), yaw, camera['yaws'], camera['fov'], camera['width'],
            camera['height'], world, observer_actor=actor,
            los_fail_closed=binding['los_fail_closed'])
        valid = checker.observation_is_confirmable(visible, distance, los, 'UGV_SIXVIEW')
        tested_goals.append([port, None if initial else yaw, initial, bool(valid)])
        if valid:
            goals[(port, yaw)] = {'distance_to_target_m': distance,
                'gt_visible': bool(visible), 'visible_views': views, 'los_clear': bool(los)}
        return valid
    def goal_test(port, yaw):
        return pose_goal(port, yaw)
    try:
        start = plan['ugv_start_port_fine_node']
        # New sparse starts must coincide with their legal logical port.
        sn = store.nodes[start]
        if math.hypot(sn.x - original.location.x, sn.y - original.location.y) > 1e-4:
            raise NavigationOracleError("oracle actual start differs from exact sparse port")
        route = shortest_confirmable_route(adapter, store.nodes, start, original.rotation.yaw,
            goal_test, binding['intersection_max_distance_m'],
            initial_goal_test=lambda p, y: pose_goal(p, y, initial=True))
        if route is not None:
            if binding['oracle_version'] in LAYERED_ORACLE_VERSIONS:
                route['tested_goal_states'] = copy.deepcopy(tested_goals)
            route['goal_evidence'] = goals.get((route['goal']['port'], route['goal']['yaw_deg']), {})
            restore_start()
            replay = replay_semantic_route(route, adapter, store.nodes, iface,
                                           binding['intersection_max_distance_m'])
            if not pose_goal(route['goal']['port'], route['goal']['yaw_deg'],
                             initial=route['goal_pose_kind'] == 'initial_spawn'):
                raise NavigationOracleError("optimal path replay cannot satisfy production CONFIRM")
            replay['final_confirmable'] = True
            route['production_replay'] = replay
        return build_certificate(route, binding)
    finally:
        restore_start()


def verify_deployed_certificate(plan, config, store, adapter, iface, target_actor,
                                distractor_actors=(), control_graph_sha='', evidence_dir=None):
    """Recompute the actual optimum before evaluation; never overwrite the plan.

    This is private preflight, with no agent steps or model inputs. Errors are
    fatal, with actual before/after bindings and a failed receipt preserved.
    """
    evidence = Path(evidence_dir) if evidence_dir is not None else None
    if evidence is None:
        raise NavigationOracleError('layered deployment requires private evidence directory')
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = {'passed': False, 'actual_oracle_recomputed': False}
    def write(name, value):
        (evidence/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    try:
        expected = plan.get('navigation_oracle')
        before = deployment_binding(plan, config, store, adapter, iface, target_actor,
                                    distractor_actors, control_graph_sha)
        write('deployment_binding_before.json', before)
        validate_certificate(expected, before)
        if expected['version'] not in LAYERED_ORACLE_VERSIONS:
            raise NavigationOracleError('actual Oracle preflight requires layered certificate')
        receipt['deployment'] = compare_deployments(expected['binding'], before)
        actual = create_deployment_certificate(plan, config, store, adapter, iface,
                                               target_actor, distractor_actors, control_graph_sha)
        receipt['actual_oracle_recomputed'] = True
        write('actual_navigation_oracle.json', actual)
        validate_certificate(actual, actual['binding'])
        compare_deployments(expected['binding'], actual['binding'])
        # All popped goal states up to the optimum, the winning goal and route
        # must reproduce. This does not claim to enumerate the whole road graph.
        for field in ('distance_m', 'goal', 'goal_pose_kind', 'actions', 'steps',
                      'fine_path', 'tested_goal_states'):
            if expected.get(field) != actual.get(field):
                raise NavigationOracleError('actual Oracle route/goal changed: '+field)
        start_a, start_b = expected.get('start', {}), actual.get('start', {})
        from decimal import Decimal
        angle_limit = expected['binding']['deployment_validation']['angle_abs_deg']
        if start_a.get('port') != start_b.get('port') or abs(_decimal_delta(
                start_a.get('yaw_deg', 0), start_b.get('yaw_deg', 0), True)) > Decimal(str(angle_limit)):
            raise NavigationOracleError('actual Oracle start changed')
        for field in ('gt_visible', 'los_clear'):
            if expected.get('goal_evidence', {}).get(field) != actual.get('goal_evidence', {}).get(field):
                raise NavigationOracleError('actual Oracle legal Confirm evidence changed: '+field)
        if not actual.get('production_replay', {}).get('final_confirmable'):
            raise NavigationOracleError('actual Oracle has no verified legal Confirm endpoint')
        after = deployment_binding(plan, config, store, adapter, iface, target_actor,
                                   distractor_actors, control_graph_sha)
        write('deployment_binding_after.json', after)
        receipt['after'] = compare_deployments(expected['binding'], after)
        receipt.update(passed=True, frozen_certificate_sha256=expected['certificate_sha256'],
                       actual_certificate_sha256=actual['certificate_sha256'],
                       frozen_distance_m=expected['distance_m'], actual_distance_m=actual['distance_m'])
        return actual
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write('deployment_validation_receipt.json', receipt)
