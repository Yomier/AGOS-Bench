"""Private Oracle probes bound to predecision observations, never agent inputs."""
import math
from agos.perception.perception import check_visibility_6view, check_visibility_unified

class _RaycastAuditWorld:
    """Observe swallowed fail-open failures; preserve the underlying LOS behavior.

    No perception/SuccessChecker source or certificate bindings are changed.
    The metric alone becomes invalid when the Oracle measurement failed.
    """
    def __init__(self, world):
        self._world = world
        self.errors = []

    def __getattr__(self, name):
        return getattr(self._world, name)

    def cast_ray(self, *args, **kwargs):
        try:
            result = self._world.cast_ray(*args, **kwargs)
        except Exception as err:
            self.errors.append('%s: %s' % (type(err).__name__, err))
            raise
        if result is None:
            self.errors.append('raycast_unavailable')
        return result

def actor_xy(actor):
    if actor is None or not actor.is_alive:
        return None
    try:
        loc = actor.get_location()
        xy = (float(loc.x), float(loc.y))
        return xy if all(math.isfinite(v) for v in xy) else None
    except Exception:
        return None

def _visibility_error(info):
    if isinstance(info, dict):
        if info.get('reason') in ('error', 'target_unavailable', 'invalid_frame',
                                  'raycast_unavailable', 'los_raycast_error_fail_closed',
                                  'los_raycast_error_fail_open') or info.get('error'):
            return True
        return any(_visibility_error(v) for v in info.values() if isinstance(v, (list, dict)))
    if isinstance(info, list):
        return any(_visibility_error(v) for v in info)
    return False

# task63: private raw-LOS diagnostics. These helpers only copy values already
# carried by the real camera packet / world handle; they never raise and never
# take part in the valid/visible decision.
def _uav_frame_meta(frame):
    meta = {'status': 'ok'}
    try:
        meta['frame_id'] = getattr(frame, 'frame_id', None)
    except Exception:
        meta['frame_id'] = None
    try:
        timestamp = getattr(frame, 'timestamp_s', None)
        meta['timestamp_s'] = float(timestamp) if timestamp is not None else None
    except Exception:
        meta['timestamp_s'] = None
    for key in ('width', 'height'):
        try:
            value = getattr(frame, key, None)
            meta[key] = int(value) if value is not None else None
        except Exception:
            meta[key] = None
    for key in ('x', 'y', 'z', 'yaw_deg', 'pitch_deg', 'fov_deg'):
        try:
            value = getattr(frame, key, None)
            meta[key] = float(value) if value is not None else None
        except Exception:
            meta[key] = None
    try:
        meta['image_bytes_len'] = len(frame.image_bytes or b'')
    except Exception:
        meta['image_bytes_len'] = None
    return meta

def _uav_world_snapshot_meta(world):
    meta = {'status': 'ok', 'frame': None, 'elapsed_seconds': None}
    try:
        snapshot = world.get_snapshot()
    except Exception as err:
        meta['status'] = 'unavailable'
        meta['detail'] = '%s: %s' % (type(err).__name__, err)
        return meta
    try:
        frame = getattr(snapshot, 'frame', None)
        meta['frame'] = int(frame) if frame is not None else None
    except Exception:
        meta['frame'] = None
    try:
        timestamp = getattr(snapshot, 'timestamp', None)
        elapsed = getattr(timestamp, 'elapsed_seconds', None)
        meta['elapsed_seconds'] = float(elapsed) if elapsed is not None else None
    except Exception:
        meta['elapsed_seconds'] = None
    return meta

def probe_ugv_observation(state, target, config, iface, checker, views=None):
    """Snapshot before decide/MOVE; all values stay in evaluator-local storage."""
    pose = dict(state.ugv_pos)
    snapshot = {'step': state.step, 'pose_xy': [pose['x'], pose['y']],
                'yaw': pose.get('yaw', 0), 'valid': False, 'visible': False,
                'visible_views': [], 'debug': {}, 'los_clear': False,
                'distance_m': None, 'confirmable': False}
    xy = actor_xy(target)
    if xy is None:
        snapshot['error'] = 'target_xy_unavailable'
        return snapshot
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (*snapshot['pose_xy'], snapshot['yaw'])):
        snapshot['error'] = 'invalid_observer_pose'
        return snapshot
    snapshot['distance_m'] = math.hypot(pose['x']-xy[0], pose['y']-xy[1])
    cfg = config.get('ugv', {}) or {}
    try:
        world = _RaycastAuditWorld(iface.get_world())
        vis, ids, debug, los = check_visibility_6view(
            target, tuple(snapshot['pose_xy']), snapshot['yaw'], cfg['stop_observe_yaws'],
            cfg.get('stop_camera_fov_deg', 90), cfg.get('stop_panel_width', 480),
            cfg.get('stop_panel_height', 270), world,
            observer_actor=getattr(iface, '_ugv_actor', None),
            los_fail_closed=bool((config.get('perception', {}) or {}).get('los_fail_closed', False)))
        snapshot.update(visible=bool(vis), visible_views=ids, debug=debug, los_clear=los)
        frames_ok = views is None or (bool(views) and all(
            getattr(v, 'valid', False) and bool(getattr(v, 'image_bytes', b'')) for v in views))
        snapshot['valid'] = frames_ok and not _visibility_error(debug) and not world.errors
        if world.errors:
            snapshot['raycast_errors'] = world.errors
        snapshot['confirmable'] = snapshot['valid'] and checker.observation_is_confirmable(
            vis, snapshot['distance_m'], los, 'UGV_SIXVIEW')
        if not snapshot['valid']:
            snapshot['error'] = 'invalid_rgb_or_visibility_probe'
    except Exception as err:
        snapshot['error'] = '%s: %s' % (type(err).__name__, err)
    return snapshot

def record_ugv_observation(state, snapshot):
    state.gva_sampling_valid = getattr(state, 'gva_sampling_valid', True) and snapshot['valid']
    if not snapshot['valid']:
        errors = getattr(state, 'metric_sampling_errors', [])
        errors.append({'metric': 'GVA', 'step': snapshot['step'], 'error': snapshot.get('error')})
        state.metric_sampling_errors = errors
        return
    distance = snapshot['distance_m']
    prev = getattr(state, 'min_ugv_target_distance_m', None)
    state.min_ugv_target_distance_m = distance if prev is None else min(prev, distance)
    if snapshot['visible']:
        state.gt_visible_in_stop6 = True
        prev = getattr(state, 'min_visible_target_distance_m', None)
        state.min_visible_target_distance_m = distance if prev is None else min(prev, distance)
    if snapshot['confirmable']:
        state.osr = True
        if not state.oracle_verifiable_reached:
            state.oracle_verifiable_reached = True
            state.first_oracle_verifiable_step = snapshot['step']
            state.first_predecision_oracle_verifiable_step = snapshot['step']

def probe_uav_observation(target, frame, world, los_fail_closed):
    if target is None or not target.is_alive or frame is None or not frame.valid or not frame.image_bytes:
        return {'valid': False, 'visible': False, 'error': 'target_or_sensor_frame_unavailable'}
    # task63: private frame-identity / world-snapshot diagnostics. Capture only;
    # any failure is recorded as a local detail and cannot weaken valid/visible.
    probe_meta = {'frame': None, 'world_snapshot_before': None, 'world_snapshot_after': None}
    try:
        audited_world = _RaycastAuditWorld(world)
        try:
            probe_meta['frame'] = _uav_frame_meta(frame)
        except Exception:
            probe_meta['frame'] = {'status': 'unavailable'}
        try:
            probe_meta['world_snapshot_before'] = _uav_world_snapshot_meta(world)
        except Exception:
            probe_meta['world_snapshot_before'] = {'status': 'unavailable'}
        visible, info = check_visibility_unified(
            target, (frame.x, frame.y, frame.z), frame.yaw_deg, frame.pitch_deg,
            frame.fov_deg, frame.width, frame.height, audited_world, los_fail_closed=los_fail_closed)
        try:
            probe_meta['world_snapshot_after'] = _uav_world_snapshot_meta(world)
        except Exception:
            probe_meta['world_snapshot_after'] = {'status': 'unavailable'}
        valid = not _visibility_error(info) and not audited_world.errors
        return {'valid': valid, 'visible': bool(visible) if valid else False, 'debug': info,
                'raycast_errors': audited_world.errors, 'probe_meta': probe_meta}
    except Exception as err:
        return {'valid': False, 'visible': False, 'error': '%s: %s' % (type(err).__name__, err),
                'probe_meta': probe_meta}
