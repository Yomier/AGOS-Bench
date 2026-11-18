# -*- coding: utf-8 -*-
"""Menu oracle behavior with an in-memory action menu."""
import math
from types import SimpleNamespace

import pytest

from agos.methods.registry import create, resolve_method_id
from agos.methods.menu_oracle.method import MenuOracleMethod, _MOVEMENT_ACTIONS


def _menu(items):
    """items: list of (action, tx, ty)"""
    return [{"action": a, "description": "d", "distance_m": 10.0,
             "target_xy": (tx, ty)} for a, tx, ty in items]


def _ugv_obs(menu=None, ux=0.0, uy=0.0, mode="UGV_SIXVIEW", active=None, cur_node=""):
    return SimpleNamespace(
        role="UGV", step=1, agent_position=(ux, uy, 0.5),
        observation_mode=mode, available_waypoints=menu or [],
        active_candidate=active, candidates=[],
        current_fine_node_id=cur_node)


def _uav_obs(frozen=False, active=None, pos=(0.0, 0.0, 50.0)):
    return SimpleNamespace(
        role="UAV", step=1, agent_position=pos, uav_frozen=frozen,
        available_waypoints=[], active_candidate=active, candidates=[])


def _cand(status="unverified", cx=40.0, cy=0.0):
    return SimpleNamespace(candidate_id="c1", status=status,
                           region_center_xy=(cx, cy))


def test_registry_creates_menu_oracle():
    mid, _prof = resolve_method_id("menu_oracle")
    assert mid == "menu_oracle"
    m = create("menu_oracle", config={})
    assert isinstance(m, MenuOracleMethod)
    assert m.method_id == "menu_oracle"


def test_ugv_gt_greedy_prefers_nearest_target():
    m = MenuOracleMethod(config={})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 90.0, "y": 0.0}}}})
    menu = _menu([("FORWARD_1", 10.0, 0.0), ("FORWARD_2", 25.0, 0.0),
                  ("TURN_LEFT", 5.0, -30.0)])
    seen = set()
    for _ in range(200):  # 覆盖牵引/随机下仍应显著偏好接近 GT 的 FORWARD 动作
        a = m._decide_ugv(_ugv_obs(menu=menu)).action
        seen.add(a)
    assert "TURN_LEFT" in seen  # 覆盖牵引保证远端动作也被触发过
    acts = []
    for _ in range(100):
        m2_seen = m._decide_ugv(_ugv_obs(menu=[x for x in menu if x["action"] != "TURN_LEFT"])).action
        acts.append(m2_seen)
    # 覆盖牵引候选耗尽后贪心主导：落点更近 GT 的 FORWARD_2(25m) 应占多数
    assert acts.count("FORWARD_2") > acts.count("FORWARD_1")


def test_reverse_only_via_coverage_pull_not_greedy():
    m = MenuOracleMethod(config={})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 100.0, "y": 0.0}}}})
    menu = _menu([("FORWARD_1", 30.0, 0.0), ("REVERSE", -20.0, 0.0)])
    picks = [m._decide_ugv(_ugv_obs(menu=menu)).action for _ in range(50)]
    assert "REVERSE" in picks          # 覆盖牵引会触发
    assert picks.count("FORWARD_1") > picks.count("REVERSE")  # 贪心主导


def test_candidate_lifecycle_actions():
    m = MenuOracleMethod(config={"success": {"max_confirm_distance_m": 50}})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 40.0, "y": 0.0}}}})
    # 充分接近(<=25m) -> 直接 CONFIRM（带 active cid）
    a0 = m._decide_ugv(_ugv_obs(ux=20.0, active=_cand("assigned_to_ugv")))
    assert a0.action == "CONFIRM" and a0.candidate_id == "c1"
    # 距 GT 40m 有更近落点(15m) -> 移动接近而非确认
    a1 = m._decide_ugv(_ugv_obs(menu=_menu([("FORWARD_1", 15.0, 0.0)]),
                                active=_cand("unverified")))
    assert a1.action == "FORWARD_1"
    # 距 GT 40m 无更近落点(90,0 距 GT 50m) 但 <=max_confirm -> 局部最优直接确认
    a2 = m._decide_ugv(_ugv_obs(menu=_menu([("FORWARD_1", 90.0, 0.0)]),
                                active=_cand("assigned_to_ugv")))
    assert a2.action == "CONFIRM" and a2.candidate_id == "c1"
    # 距 GT 40m 无更近落点且无候选 -> 独立确认（cid=None）
    a3 = m._decide_ugv(_ugv_obs(menu=_menu([("FORWARD_1", 90.0, 0.0)])))
    assert a3.action == "CONFIRM" and not a3.candidate_id


def test_no_candidate_independent_discovery():
    m = MenuOracleMethod(config={})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 20.0, "y": 0.0}}}})
    # 距 GT <=25m 且菜单无可再接近落点/无菜单 -> CONFIRM；先看 10m 直接确认
    a = m._decide_ugv(_ugv_obs(ux=10.0, menu=_menu([("FORWARD_1", 25.0, 0.0)])))
    assert a.action == "CONFIRM"


def test_empty_menu_waits_not_crash():
    m = MenuOracleMethod(config={})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 90.0, "y": 0.0}}}})
    a = m._decide_ugv(_ugv_obs(menu=[]))
    assert a.action == "WAIT"


def test_uav_frozen_waits_and_annotates_when_free():
    m = MenuOracleMethod(config={})
    m.start_episode({"episode_plan": {"target_oracle": {"transform": {"x": 5.0, "y": 5.0}}}})
    a = m._decide_uav(_uav_obs(frozen=True))
    assert a.action == "WAIT"
    from agos.agents.agents import validate_action
    from agos.lite_core import LiteObservation

    observation = LiteObservation("example", 1, 80, "UAV", agent_position=(0.0, 0.0, 50.0),
                                  available_actions=["SUSPECT_CANDIDATE", "WAIT"], max_annotations=1)
    annotation = m._decide_uav(observation)
    assert annotation.action == "ANNOTATE_CANDIDATE"
    assert annotation.geometry_type == "point" and annotation.grid_cell is None
    assert tuple(annotation.position_xy) == (5.0, 5.0)
    valid, reason = validate_action(vars(annotation), observation, set())
    assert valid, reason


def test_movement_actions_constant_matches_runner_six():
    assert _MOVEMENT_ACTIONS == ("FORWARD_1", "FORWARD_2", "TURN_LEFT",
                                 "TURN_RIGHT", "TURN_STRAIGHT", "REVERSE")


def test_certified_route_guided_navigation():
    """menu_oracle 2.0：证书路线序号贪心（沿路线快进 + 到 approach 确认）。"""
    m = MenuOracleMethod(config={"success": {"max_confirm_distance_m": 50}})
    plan = {"target_oracle": {"transform": {"x": 300.0, "y": 0.0}},
            "static_solvability_precheck": {
                "certified_route_node_ids": ["n0", "n1", "n2", "n3", "n4"],
                "certified_ugv_approach_node_id": "n4"}}
    m.start_episode({"episode_plan": plan})
    # 当前 n0：菜单 F1->n1 / F2->n2 / REVERSE->n_1(不在路线)；序号贪心选 F2（快进）
    menu = [{"action": "FORWARD_1", "description": "", "distance_m": 20.0,
             "target_xy": (20.0, 0.0), "target_fine_node_id": "n1"},
            {"action": "FORWARD_2", "description": "", "distance_m": 40.0,
             "target_xy": (40.0, 0.0), "target_fine_node_id": "n2"},
            {"action": "REVERSE", "description": "", "distance_m": 20.0,
             "target_xy": (-20.0, 0.0), "target_fine_node_id": "n_1"}]
    a = m._decide_ugv(_ugv_obs(menu=menu, cur_node="n0"))
    assert a.action == "FORWARD_2"
    # 到达 approach node -> 直接 CONFIRM（无论距离）
    a2 = m._decide_ugv(_ugv_obs(menu=menu, cur_node="n4"))
    assert a2.action == "CONFIRM"


def test_certified_route_never_early_confirms_on_distance_only():
    """R12 regression: a certified route must reach its bound approach before CONFIRM."""
    m = MenuOracleMethod(config={"success": {"max_confirm_distance_m": 50}})
    plan = {
        "target_oracle": {"transform": {"x": 25.4, "y": 191.6}},
        "static_solvability_precheck": {
            "certified_route_node_ids": ["n0", "n1", "n2", "n3", "n4"],
            "certified_route_actions": ["REVERSE", "FORWARD_1", "TURN_STRAIGHT", "REVERSE"],
            "certified_ugv_approach_node_id": "n4",
        },
    }
    m.start_episode({"episode_plan": plan})
    # n2 is only ~22m from GT (the exact R12 failure pattern), but n2 is not
    # the certified approach. The oracle must continue with TURN_STRAIGHT.
    menu = [{"action": "TURN_STRAIGHT", "description": "",
             "distance_m": 50.0, "target_xy": (60.0, 191.6),
             "target_fine_node_id": "n3"}]
    a = m._decide_ugv(_ugv_obs(menu=menu, ux=3.4, uy=191.6, cur_node="n2",
                                active=_cand("assigned_to_ugv")))
    assert a.action == "TURN_STRAIGHT"


def test_certified_route_mismatch_waits_fail_closed():
    m = MenuOracleMethod(config={})
    plan = {
        "target_oracle": {"transform": {"x": 10.0, "y": 0.0}},
        "static_solvability_precheck": {
            "certified_route_node_ids": ["n0", "n1"],
            "certified_route_actions": ["FORWARD_1"],
            "certified_ugv_approach_node_id": "n1",
        },
    }
    m.start_episode({"episode_plan": plan})
    # No legal menu item matches the certified transition; do not silently
    # fall back to a distance-based CONFIRM.
    a = m._decide_ugv(_ugv_obs(menu=[], ux=0.0, uy=0.0, cur_node="n0"))
    assert a.action == "WAIT"
