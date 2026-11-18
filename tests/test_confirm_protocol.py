"""30m benchmark semantics and scientific aggregation contracts."""
from types import SimpleNamespace as NS
import copy
import pytest

from agos.eval.eval import SuccessChecker, Metrics

def cfg(radius=None):
    result = {'success': {'require_stop_observe': False,
              'require_gt_visible_in_stop_6view': True, 'require_los': True},
              'ugv': {'confirm_budget': 1}}
    if radius is not None:
        result['success']['max_confirm_distance_m'] = radius
    return result


def test_new_default_is_30m_and_explicit_legacy_50m_works():
    assert SuccessChecker(cfg()).max_confirm_dist == 30.0
    assert SuccessChecker(cfg(50)).max_confirm_dist == 50.0


@pytest.mark.parametrize('distance,expected', [(29.99, True), (30, True), (30.01, False), (50, False)])
def test_checker_inclusive_boundary_keeps_terminal_false_confirm(distance, expected):
    state = NS(confirm_count=0, confirm_budget=1, ugv_observation_mode='UGV_SIXVIEW',
               false_confirm_count=0, terminal=False, success=False, osr=False)
    ok, reason = SuccessChecker(cfg(30)).check(state, NS(action='CONFIRM'), True, distance, True)
    assert ok is expected
    assert state.terminal is True
    assert state.confirm_budget == 0
    assert state.false_confirm_count == int(not expected)


@pytest.mark.parametrize('los', [False, None])
def test_los_still_fails_at_20m(los):
    checker = SuccessChecker(cfg(30))
    assert not checker.observation_is_confirmable(True, 20, los)


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf'), True, 'nonsense'])
def test_bad_distance_cannot_produce_plausible_checker(radius):
    with pytest.raises(ValueError):
        SuccessChecker(cfg(radius))


def test_mixed_success_radii_cannot_aggregate_even_with_same_spl_oracle():
    metrics = Metrics()
    for radius in (30, 50):
        metrics.add_episode({'success': False, 'max_confirm_distance_m': radius,
                             'spl_oracle_version': 'semantic_meters_confirm_v1',
                             'UGV_SPL_contribution': 0})
    with pytest.raises(ValueError, match='mixed.*success|success.*protocol'):
        metrics.compute_joint_metrics()


def test_ugv_only_also_rejects_mixed_protocols():
    metrics = Metrics()
    metrics.episodes = [{'success': False, 'max_confirm_distance_m': r} for r in (30, 50)]
    with pytest.raises(ValueError, match='mixed.*success|success.*protocol'):
        metrics.compute_ugv_search_only_metrics()


def test_full_protocol_mixing_catches_los_changes_at_same_radius():
    from agos.benchmark_protocol import success_protocol
    first = cfg(30)
    second = copy.deepcopy(first)
    second['success']['require_los'] = False
    metrics = Metrics()
    metrics.episodes = [{'success': False, 'success_protocol': success_protocol(c)}
                        for c in (first, second)]
    with pytest.raises(ValueError, match='mixed.*success|success.*protocol'):
        metrics.compute_joint_metrics()


def test_explicit_protocol_tag_cannot_silently_conflict_with_radius():
    from agos.benchmark_protocol import success_protocol
    config = cfg(50)
    config['benchmark'] = {'success_protocol_id': 'AGOS_CONFIRM30_V1'}
    with pytest.raises(ValueError, match='protocol'):
        success_protocol(config)


def test_protocol_change_never_qualifies_for_settle_noise_rebind():
    from agos.runtime.runner import _rebind_navigation_certificate
    old = {'binding': {'success': {'max_confirm_distance_m': 50}}, 'distance_m': 100}
    assert _rebind_navigation_certificate(old, {'success': {'max_confirm_distance_m': 30}}) is None


def test_radius_only_record_cannot_disagree_with_full_metadata():
    from agos.benchmark_protocol import success_protocol
    metrics = Metrics()
    metrics.episodes = [{'success': False, 'max_confirm_distance_m': 50,
                         'success_protocol': success_protocol(cfg(30))}]
    with pytest.raises(ValueError, match='protocol|radius'):
        metrics.compute_joint_metrics()


def test_default_point_pool_excludes_outer_50m_node():
    from agos.roadnet.candidate_approach import compute_point_service_nodes
    store = NS(nodes={'far': NS(x=40, y=0), 'near': NS(x=29, y=0)})
    pool, band = compute_point_service_nodes(store, (0, 0))
    assert pool == frozenset({'near'})
    assert band == 30
    pool, band = compute_point_service_nodes(NS(nodes={'far': NS(x=40, y=0)}), (0, 0))
    assert not pool and band is None

def test_protocol_preflight_rejects_old_task_before_simulator():
    from agos.benchmark_protocol import validate_plan_success_protocol
    plan = {'navigation_oracle': {'binding': {'success': cfg(50)['success']}}}
    with pytest.raises(ValueError, match='recertification'):
        validate_plan_success_protocol(plan, cfg(30))
