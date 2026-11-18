"""Effective success protocol, shared by evaluator, tools, prompts and metadata.

30m is the current default. Explicit 50m configurations remain a legacy protocol;
frozen data/certificates must never be silently relabelled as the current one.
"""
import json
import math

DEFAULT_CONFIRM_DISTANCE_M = 30.0
DISTANCE_METRIC = 'target_actor_center_xy_m'


def confirm_distance_m(config):
    raw = (config.get('success', {}) or {}).get('max_confirm_distance_m', DEFAULT_CONFIRM_DISTANCE_M)
    try:
        radius = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError('success protocol requires a finite positive confirm radius') from error
    if isinstance(raw, bool) or not math.isfinite(radius) or radius <= 0:
        raise ValueError('success protocol requires a finite positive confirm radius')
    return radius


def format_confirm_distance(config):
    return format(confirm_distance_m(config), 'g')


def generation_approach_min_m(config, legacy_default_m=25.0):
    radius = confirm_distance_m(config)
    raw = (config.get('generation', {}) or {}).get('min_approach_distance_m',
        legacy_default_m if radius == 50.0 else 5.0)
    minimum = float(raw)
    if isinstance(raw, bool) or not math.isfinite(minimum) or not 0 < minimum <= radius:
        raise ValueError('generation.min_approach_distance_m must be positive and <= confirmation distance')
    return minimum


def success_protocol(config):
    success = config.get('success', {}) or {}
    radius = confirm_distance_m(config)
    protocol_id = 'AGOS_CONFIRM%s_V1' % format(radius, 'g')
    tag = (config.get('benchmark', {}) or {}).get('success_protocol_id')
    if tag is not None and tag != protocol_id:
        raise ValueError('success protocol ID %s conflicts with effective radius %sm' % (tag, radius))
    return {'id': protocol_id, 'max_confirm_distance_m': radius,
        'require_stop_observe': bool(success.get('require_stop_observe', False)),
        'require_gt_visible_in_stop_6view': bool(success.get('require_gt_visible_in_stop_6view', True)),
        'require_los': bool(success.get('require_los', False)),
        'los_fail_closed': bool((config.get('perception', {}) or {}).get('los_fail_closed', False)),
        'distance_metric': DISTANCE_METRIC}


def apply_success_protocol(config, radius_m=DEFAULT_CONFIRM_DISTANCE_M):
    # An explicit migration operation, never a loader silently overriding 50m.
    config.setdefault('success', {})['max_confirm_distance_m'] = radius_m
    config.setdefault('benchmark', {}).pop('success_protocol_id', None)
    protocol = success_protocol(config)
    config['benchmark']['success_protocol_id'] = protocol['id']
    return config


def validate_plan_success_protocol(plan, config):
    """Cheap preflight before actors/models; real deployment binding still required."""
    expected = success_protocol(config)
    declared = plan.get('success_protocol')
    binding = (plan.get('navigation_oracle') or {}).get('binding', {}) or {}
    cert_success = binding.get('success')
    if declared is not None and declared != expected:
        raise ValueError('task success protocol mismatch: recertify into a separate dataset; do not relabel the old certificate')
    if cert_success is not None:
        old = success_protocol({'success': cert_success,
                 'perception': {'los_fail_closed': binding.get('los_fail_closed', False)}})
        if old != expected:
            raise ValueError('navigation certificate success protocol mismatch (%s -> %s): '
                'real deployment recertification and a new SPL optimum are required' % (old['id'], expected['id']))


def assert_uniform_result_protocol(episodes):
    """Fail instead of combining different success definitions into SR/SPL."""
    keys = set()
    for result in episodes:
        metadata = result.get('success_protocol')
        radius = result.get('max_confirm_distance_m', result.get('success_max_confirm_distance_m'))
        if metadata is not None:
            if not isinstance(metadata, dict) or 'max_confirm_distance_m' not in metadata:
                raise ValueError('invalid success protocol result metadata')
            effective = success_protocol({'success': metadata,
                'benchmark': {'success_protocol_id': metadata.get('id')},
                'perception': {'los_fail_closed': metadata.get('los_fail_closed', False)}})
            if metadata != effective:
                raise ValueError('incomplete or inconsistent success protocol result metadata')
            if radius is not None and confirm_distance_m({'success': {'max_confirm_distance_m': radius}}) != effective['max_confirm_distance_m']:
                raise ValueError('result radius disagrees with success protocol metadata')
            keys.add(json.dumps(effective, sort_keys=True))
        elif radius is not None:
            # Historical radius-only records do not prove LOS/visibility settings.
            keys.add('legacy_radius_only:%s' % format_confirm_distance({'success': {'max_confirm_distance_m': radius}}))
        else:
            keys.add('legacy_unknown')
    if len(keys) > 1:
        raise ValueError('mixed success protocols cannot aggregate SR/SPL; group results by complete protocol')
    return sorted(keys)
