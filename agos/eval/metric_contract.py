"""Evaluation-only metric generation and terminal XY navigation error."""
import math
from numbers import Real

METRIC_REVISION = 'AGOS_METRICS_V6_OSR_GVA_ACTION_20261013'
V5_PROTOCOL = {'id': 'AGOS_METRICS_V5_TERMINAL_NE_OBS_20261013',
               'ne': 'terminal_ugv_target_xy_m',
               'ase': 'first_valid_uav_observation_fixed_budget',
               'acg': 'v3_rank_polygon_distance_scale50_halfsize15',
               'gva': 'v3_predecision_opportunity_or_invalid_confirm'}
ACG_POINT_HALF_SIZE_M = 15.0
_UNSPECIFIED_POSE = object()

def metric_protocol():
    return {'id': METRIC_REVISION, 'ne': 'terminal_ugv_target_xy_m',
            'ase': 'first_valid_uav_observation_fixed_budget',
            'acg': 'v3_rank_polygon_distance_scale50_halfsize15',
            'osr': 'predecision_legal_confirm_opportunity',
            'gva': 'confirm_reject_action_micro_accuracy'}

def assert_uniform_metric_protocol(episodes):
    ids = set()
    for ep in episodes:
        metadata = ep.get('metric_protocol')
        if metadata is None:
            ids.add('legacy_unversioned')
        elif metadata == V5_PROTOCOL:
            ids.add(V5_PROTOCOL['id'])
        elif metadata != metric_protocol():
            raise ValueError('unsupported or inconsistent metric protocol')
        else:
            ids.add(METRIC_REVISION)
    if len(ids) > 1:
        raise ValueError('mixed metric protocols cannot aggregate; separate old and new results')
    return sorted(ids)

def aggregate_osr(episodes):
    valid = all(type(e.get('oracle_verifiable_reached')) is bool
                and e.get('gva_sampling_valid') is True
                and (not e.get('success') or e['oracle_verifiable_reached']) for e in episodes)
    reached = sum(e.get('oracle_verifiable_reached') is True for e in episodes)
    return {'OSR': reached/len(episodes) if valid and episodes else None,
            'osr_aggregation_valid': valid, 'osr_support': len(episodes),
            'osr_opportunity_count': reached,
            'osr_definition': 'predecision_legal_confirm_opportunity'}

def aggregate_verification_actions(episodes):
    """Micro accuracy of executed judgment actions, not candidate identity accuracy.

    No verdict is not a false negative. Broken/missing Oracle labels invalidate
    the aggregate rather than being treated as negative ground truth.
    """
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    complete = True
    episode_support = 0
    invalid_count = 0
    for ep in episodes:
        events = ep.get('gva_events')
        if not isinstance(events, list):
            complete = False
            continue
        if ep.get('gva_sampling_valid') is not True:
            complete = False
        if events:
            episode_support += 1
        for event in events:
            if (not isinstance(event, dict) or event.get('oracle_valid') is not True
                    or type(event.get('oracle_positive')) is not bool
                    or event.get('model_verdict') not in ('CONFIRM', 'REJECT')):
                complete = False
                invalid_count += 1
                continue
            positive = event['oracle_positive']
            confirm = event['model_verdict'] == 'CONFIRM'
            counts['tp' if confirm and positive else 'fp' if confirm else 'fn' if positive else 'tn'] += 1
    n = sum(counts.values())
    return {'GVA': (counts['tp']+counts['tn'])/n if complete and n else None,
            'gva_aggregation_valid': complete, 'gva_support': n,
            'gva_episode_support': episode_support, 'gva_invalid_event_count': invalid_count,
            'gva_definition': 'confirm_reject_action_micro_accuracy',
            'gva_positive_support': counts['tp']+counts['fn'],
            'gva_negative_support': counts['tn']+counts['fp'],
            **{'gva_'+k:v for k,v in counts.items()}}

def finite_distance(value):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)) and value >= 0)

def finalize_navigation_error(state, target_xy, terminal_xy=_UNSPECIFIED_POSE):
    """Use terminal executed pose, never a stale Confirm distance or planned target.

    target_xy must be the actual deployed actor XY, read before cleanup. If the
    actor cannot be read, pass None and surface invalid NE; no GT-plan fallback.
    """
    try:
        xy = (state.ugv_pos['x'], state.ugv_pos['y']) if terminal_xy is _UNSPECIFIED_POSE else terminal_xy
        coords = (xy[0], xy[1], target_xy[0], target_xy[1])
        valid = all(isinstance(v, Real) and not isinstance(v, bool)
                    and math.isfinite(float(v)) for v in coords)
    except (TypeError, KeyError, IndexError, AttributeError):
        valid = False
    distance = math.hypot(coords[0]-coords[2], coords[1]-coords[3]) if valid else None
    confirm = getattr(state, 'confirm_distance_m', None)
    # SuccessChecker keeps its frozen success semantics and code binding. Its
    # pre-v5 internal ne_distance is the last actual Confirm distance; copy it
    # before terminal finalization overwrites the value, only when Confirm ran.
    if confirm is None and getattr(state, 'confirm_count', 0) > 0:
        confirm = getattr(state, 'ne_distance', None)
    state.confirm_distance_m = confirm if finite_distance(confirm) else None
    state.ne_distance = distance
    return {'ne_distance_m': distance, 'ne_valid': valid,
            'ne_invalid_reason': None if valid else 'terminal_or_target_xy_unavailable',
            'confirm_distance_m': confirm if finite_distance(confirm) else None,
            'ne_terminal_ugv_xy': list(coords[:2]) if valid else None,
            'ne_terminal_target_xy': list(coords[2:]) if valid else None}

def aggregate_navigation_error(episodes, current):
    valid_eps = [e for e in episodes if finite_distance(e.get('ne_distance_m'))
                 and e.get('ne_valid', True) is True]
    complete = len(valid_eps) == len(episodes) and bool(episodes)
    def average(items):
        return sum(float(e['ne_distance_m']) for e in items)/len(items) if items else None
    return {'avg_NE_m': average(valid_eps) if complete or not current else None,
            'avg_NE_success_m': average([e for e in valid_eps if e.get('success')]) if complete or not current else None,
            'avg_NE_failure_m': average([e for e in valid_eps if not e.get('success')]) if complete or not current else None,
            'NE_valid_samples': len(valid_eps), 'NE_missing_or_invalid_samples': len(episodes)-len(valid_eps),
            'ne_aggregation_valid': complete,
            'ne_semantics': 'terminal_ugv_target_xy_m' if current else 'legacy_confirm_subset_not_terminal_ne'}
