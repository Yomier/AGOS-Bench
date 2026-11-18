"""Distances, evidence timing and metric aggregation."""
import math
from types import SimpleNamespace as NS
import pytest

def ep(**kw):
    from agos.eval.metric_contract import metric_protocol
    result = dict(success=False, steps=3, step_logs=[], metric_protocol=metric_protocol(),
                  ne_distance_m=40., ne_valid=True, oracle_verifiable_reached=False,
                  gva_sampling_valid=True, gva_events=[], ase_sampling_valid=True, ACG_episode=0.,
                  ase_first_oracle_visible_uav_step=None, ase_max_uav_decision_steps=80,
                  aerial_metrics_applicable=True)
    result.update(kw)
    return result

def aggregate(*episodes):
    from agos.eval.eval import Metrics
    obj = Metrics()
    for item in episodes:
        obj.add_episode(item)
    return obj.compute_joint_metrics()

@pytest.mark.parametrize('confirm_count,confirm_distance', [(0, None), (1, 7.), (1, 80.)])
def test_terminal_ne_always_uses_final_xy_and_keeps_confirm_distance(confirm_count, confirm_distance):
    from agos.eval.metric_contract import finalize_navigation_error
    state = NS(ugv_pos={'x': 3., 'y': 4., 'z': 999}, ne_distance=confirm_distance,
               confirm_distance_m=confirm_distance, confirm_count=confirm_count)
    fields = finalize_navigation_error(state, (0., 0.))
    assert fields['ne_distance_m'] == 5.
    assert fields['confirm_distance_m'] == confirm_distance
    assert fields['ne_valid'] is True

def test_terminal_ne_uses_unrounded_coordinates():
    from agos.eval.metric_contract import finalize_navigation_error
    state = NS(ugv_pos={'x': .03, 'y': .04}, confirm_distance_m=None)
    assert finalize_navigation_error(state, (0., 0.))['ne_distance_m'] == .05

def test_ne_aggregate_includes_failed_episode_without_confirm():
    out = aggregate(ep(success=True, ne_distance_m=10., oracle_verifiable_reached=True),
                    ep(ne_distance_m=90., confirm_count=0))
    assert out['avg_NE_m'] == 50. and out['NE_valid_samples'] == 2
    assert out['avg_NE_failure_m'] == 90.
    assert out['ne_aggregation_valid']

def test_no_ne_or_partial_ne_never_looks_like_zero_error():
    out = aggregate(ep(ne_distance_m=None, ne_valid=False))
    assert out['avg_NE_m'] is None and out['NE_valid_samples'] == 0
    out = aggregate(ep(ne_distance_m=5), ep(ne_distance_m=None, ne_valid=False))
    assert out['avg_NE_m'] is None and not out['ne_aggregation_valid']

@pytest.mark.parametrize('bad', [math.nan, math.inf, -1, True])
def test_nonfinite_or_negative_ne_is_not_scientific_average(bad):
    out = aggregate(ep(ne_distance_m=bad))
    assert out['avg_NE_m'] is None and not out['ne_aggregation_valid']

def test_new_and_legacy_metrics_cannot_mix():
    with pytest.raises(ValueError, match='metric'):
        aggregate(ep(), {'success': False, 'steps': 1})

def test_new_gva_missing_strict_field_does_not_fallback_to_osr():
    item = ep(osr=True)
    del item['oracle_verifiable_reached']
    out = aggregate(item)
    assert out['GVA'] is None and out['gva_strict'] is True
    assert not out['osr_aggregation_valid']

def test_gva_support_includes_invalid_confirm_and_excludes_no_opportunity():
    out = aggregate(ep(success=True, oracle_verifiable_reached=True),
                    ep(oracle_verifiable_reached=True), ep(confirm_count=1), ep())
    assert out['Verification_Opportunity_Conversion'] == pytest.approx(1/3, abs=1e-6)
    assert out['verification_opportunity_support'] == 3
    assert out['GVA'] is None and out['gva_support'] == 0

def test_nonapplicable_aerial_metrics_are_na_not_zero():
    out = aggregate(ep(aerial_metrics_applicable=False))
    assert out['ASE'] is None and out['ACG'] is None

def test_broken_visibility_probe_is_not_a_normal_miss():
    out = aggregate(ep(ase_sampling_valid=False, gva_sampling_valid=False))
    assert out['ASE'] is None and out['GVA'] is None
    assert not out['ase_aggregation_valid'] and not out['gva_aggregation_valid']

def test_acg_detail_quality_does_not_include_rank_discount():
    from agos.eval.metric_primitives import compute_acg_episode_v3
    result = compute_acg_episode_v3([{'distance_m': 1000}, {'distance_m': 0}], 80)
    assert result['best_rank'] == 2
    assert result['score'] == pytest.approx(.63093, abs=1e-6)
    assert result['best_quality'] == 1.

@pytest.mark.parametrize('tau,budget', [(0, 80), (81, 80), (True, 80), (1, 0), (1.5, 80)])
def test_ase_bad_step_not_silently_clipped(tau, budget):
    from agos.eval.metric_primitives import compute_ase_episode
    with pytest.raises(ValueError):
        compute_ase_episode(tau, budget)

@pytest.mark.parametrize('records', [[{}], [{'distance_m': -1}], [{'distance_m': math.nan}]])
def test_acg_bad_geometry_not_silently_scored(records):
    from agos.eval.metric_primitives import compute_acg_episode_v3
    with pytest.raises(ValueError):
        compute_acg_episode_v3(records, 80)

def test_gva_snapshot_is_predecision_and_not_changed_by_move(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    from agos.eval.eval import SuccessChecker
    cfg = {'success': {'require_stop_observe': False, 'require_gt_visible_in_stop_6view': True,
                      'max_confirm_distance_m': 30, 'require_los': True},
           'ugv': {'stop_observe_yaws': [0]}}
    state = NS(step=1, ugv_pos={'x': 3., 'y': 4., 'yaw': 0}, osr=False,
               oracle_verifiable_reached=False, first_oracle_verifiable_step=None,
               first_predecision_oracle_verifiable_step=None)
    target = NS(is_alive=True, get_location=lambda: NS(x=0., y=0.))
    iface = NS(get_world=lambda: None, _ugv_actor=None)
    monkeypatch.setattr(mod, 'check_visibility_6view', lambda *a, **kw: (True, [0], {}, True))
    snapshot = mod.probe_ugv_observation(state, target, cfg, iface, SuccessChecker(cfg))
    state.ugv_pos['x'] = 100  # MOVE occurs after the snapshot was taken
    mod.record_ugv_observation(state, snapshot)
    assert snapshot['distance_m'] == 5. and snapshot['pose_xy'] == [3., 4.]
    assert state.oracle_verifiable_reached and state.first_predecision_oracle_verifiable_step == 1

def test_ase_uses_actual_sensor_frame_geometry(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    frame = NS(x=3., y=4., z=73., yaw_deg=15., pitch_deg=-80., fov_deg=65.,
               width=800, height=600, valid=True, image_bytes=b'raw')
    seen = {}
    def visibility(target, pose, yaw, pitch, fov, width, height, world, **kw):
        seen.update(pose=pose, yaw=yaw, pitch=pitch, fov=fov, width=width, height=height)
        return True, {'reason': 'visible'}
    monkeypatch.setattr(mod, 'check_visibility_unified', visibility)
    result = mod.probe_uav_observation(NS(is_alive=True), frame, None, False)
    assert result['visible'] and result['valid']
    assert seen == dict(pose=(3., 4., 73.), yaw=15., pitch=-80., fov=65., width=800, height=600)

def test_missing_probe_rgb_or_target_is_explicit_invalidity(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    from agos.eval.eval import SuccessChecker
    cfg = {'success': {'require_stop_observe': False, 'require_gt_visible_in_stop_6view': True,
                      'require_los': True}, 'ugv': {'stop_observe_yaws': [0]}}
    state = NS(step=1, ugv_pos={'x': 0., 'y': 0.})
    monkeypatch.setattr(mod, 'check_visibility_6view', lambda *a, **kw: (True, [0], {}, True))
    target = NS(is_alive=True, get_location=lambda: NS(x=3., y=4.))
    result = mod.probe_ugv_observation(state, target, cfg, NS(get_world=lambda: None),
                                     SuccessChecker(cfg), views=[NS(valid=False, image_bytes=b'')])
    assert not result['valid'] and not result['confirmable']
    assert mod.probe_uav_observation(target, None, None, False)['valid'] is False

def test_confirm_checker_keeps_distance_diagnostic_separate():
    from agos.eval.eval import SuccessChecker
    from agos.eval.metric_contract import finalize_navigation_error
    cfg = {'success': {'require_stop_observe': False, 'require_gt_visible_in_stop_6view': True},
           'ugv': {'confirm_budget': 1}}
    state = NS(confirm_count=0, confirm_budget=1, false_confirm_count=0,
               ugv_pos={'x': 3., 'y': 4.})
    SuccessChecker(cfg).check(state, NS(action='CONFIRM'), False, 45., los_clear=True)
    fields = finalize_navigation_error(state, (0., 0.))
    assert fields['confirm_distance_m'] == 45. and fields['ne_distance_m'] == 5.

def test_ase_failed_probe_and_fixed_horizon_examples(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    from agos.eval.metric_primitives import compute_ase_episode
    frame = NS(x=0., y=0., z=80., yaw_deg=0., pitch_deg=-90., fov_deg=90.,
               width=1024, height=576, valid=True, image_bytes=b'raw')
    monkeypatch.setattr(mod, 'check_visibility_unified',
                        lambda *a, **kw: (False, {'reason': 'error', 'error': 'raycast failed'}))
    assert not mod.probe_uav_observation(NS(is_alive=True), frame, None, False)['valid']
    assert compute_ase_episode(5, 80) == .95
    assert compute_ase_episode(None, 80) == 0

def test_los_api_unavailable_is_not_legitimate_metric_occlusion(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    frame = NS(x=0., y=0., z=80., yaw_deg=0., pitch_deg=-90., fov_deg=90.,
               width=1024, height=576, valid=True, image_bytes=b'raw')
    monkeypatch.setattr(mod, 'check_visibility_unified', lambda *a, **kw:
                        (True, {'reason': 'visible', 'los_debug': [{'reason': 'raycast_unavailable'}]}))
    assert not mod.probe_uav_observation(NS(is_alive=True), frame, None, False)['valid']

def test_terminal_ne_prefers_actual_actor_pose_to_logical_waypoint():
    from agos.eval.metric_contract import finalize_navigation_error
    state = NS(ugv_pos={'x': 100., 'y': 100.}, confirm_count=0)
    assert finalize_navigation_error(state, (0., 0.), (3., 4.))['ne_distance_m'] == 5.
    assert finalize_navigation_error(state, (0., 0.), None)['ne_valid'] is False

def test_ugv_camera_capture_and_scoring_have_same_configured_geometry():
    from agos.runtime.runner import _capture_metric_ugv_views
    seen = {}
    iface = NS(capture_ugv_views=lambda **kw: seen.update(kw) or ())
    cfg = {'ugv': {'stop_observe_yaws': [0, 180], 'stop_camera_fov_deg': 80,
                   'stop_panel_width': 640, 'stop_panel_height': 360}}
    _capture_metric_ugv_views(iface, cfg)
    assert seen == dict(yaws=[0, 180], fov=80, panel_w=640, panel_h=360)

def test_visibility_probe_bad_pose_cannot_be_scored_as_normal_miss(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    from agos.eval.eval import SuccessChecker
    cfg = {'success': {'require_stop_observe': False, 'require_gt_visible_in_stop_6view': True},
           'ugv': {'stop_observe_yaws': [0]}}
    target = NS(is_alive=True, get_location=lambda: NS(x=0., y=0.))
    monkeypatch.setattr(mod, 'check_visibility_6view', lambda *a, **kw: (False, [], {}, False))
    snapshot = mod.probe_ugv_observation(NS(step=1, ugv_pos={'x': math.nan, 'y': 0.}),
                                       target, cfg, NS(get_world=lambda: None), SuccessChecker(cfg))
    assert not snapshot['valid']

def test_metric_probe_detects_fail_open_raycast_exception_without_changing_success_contract(monkeypatch):
    import agos.runtime.evaluation_observations as mod
    frame = NS(x=0., y=0., z=80., yaw_deg=0., pitch_deg=-90., fov_deg=90.,
               width=1024, height=576, valid=True, image_bytes=b'raw')
    def cast(*args):
        raise RuntimeError('simulator raycast unavailable')
    def swallowed_visibility(target, pose, yaw, pitch, fov, width, height, world, **kw):
        try:
            world.cast_ray(None, None)
        except RuntimeError:
            pass  # emulate frozen fail-open visibility behavior
        return True, {'reason': 'visible'}
    monkeypatch.setattr(mod, 'check_visibility_unified', swallowed_visibility)
    result = mod.probe_uav_observation(NS(is_alive=True), frame, NS(cast_ray=cast), False)
    assert not result['valid'] and not result['visible']
