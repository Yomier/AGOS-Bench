"""Configuration and public observation contracts for cooperative search."""
from dataclasses import fields

import pytest

from agos.lite_core import CandidateInfo, LiteObservation


def test_validate_config_rejects_unknown_keys():
    """A misspelling of routine_progress must fail config validation."""
    from agos.methods.cooperative_search.contracts import validate_config

    base = {"revision": "coop-event-v10", "arm": "agos_b"}
    assert validate_config({"cooperative_search": {**base, "routine_progress": 0.5}})
    with pytest.raises(ValueError, match="routine_fraction"):
        validate_config({"cooperative_search": {**base, "routine_fraction": 0.5}})


def test_context_retains_active_region_grid_without_target():
    from agos.methods.cooperative_search.contracts import public_context

    context = {
        "episode_id": "ep-public",
        "target_reference_image": b"reference",
        "target_reference_views": {"FRONT": b"reference-front"},
        "episode_plan": {
            "region": {"x_min": 0, "x_max": 80, "y_min": 0, "y_max": 80},
            "start": {"uav": {"x": 0, "y": 0, "z": 80},
                      "ugv": {"x": 5, "y": 5, "z": 0.5}},
            "target_task": {"target_description": "blue van"},
            "target_oracle": {"x": 55, "y": 55},
            "uav_grid_waypoints": [
                {"id": "U0", "x": 0, "y": 0, "active": True,
                 "target_visible": True},
                {"id": "U1", "x": 40, "y": 40, "active": True},
                {"id": "U2", "x": 80, "y": 80, "active": False},
                {"id": "U3", "x": 120, "y": 0, "active": True},
            ],
        },
    }
    public = public_context(context)

    assert public["episode_id"] == "ep-public"
    assert public["target_description"] == "blue van"
    assert public["target_reference_views"] == {"FRONT": b"reference-front"}
    assert [wp["id"] for wp in public["uav_grid_waypoints"]] == ["U0", "U1"]
    assert all("target_visible" not in wp for wp in public["uav_grid_waypoints"])
    assert "episode_plan" not in public
    assert "target_oracle" not in public
    public["start"]["ugv"]["x"] = 99
    assert context["episode_plan"]["start"]["ugv"]["x"] == 5


def test_observation_rebuild_strips_candidate_diagnostics():
    from agos.methods.cooperative_search.contracts import public_observation

    candidate = CandidateInfo("C1", "unverified", region_center_xy=(10, 20),
                              service_node_ids=("node-2",))
    candidate.gt_distance = 8.0
    candidate.gt_visible = True
    observation = LiteObservation("ep", 1, 80, "UGV", candidates=[candidate],
                                  active_candidate=candidate)
    public = public_observation(observation)

    assert public.candidates[0].service_node_ids == ("node-2",)
    assert public.active_candidate.candidate_id == "C1"
    assert "gt_distance" not in vars(public.candidates[0])
    assert "gt_visible" not in vars(public.active_candidate)
    public.candidates[0].region_polygon.append((9, 9))
    assert candidate.region_polygon == []


def test_transition_strips_gt_and_keeps_public_execution_times():
    from agos.methods.cooperative_search.contracts import public_transition

    result = {
        "status": "ok",
        "executed_action": {"role": "UGV", "action": "FORWARD_1"},
        "effect_at": 10.0,
        "ready_at": 16.0,
        "gt_visible": True,
        "dist_to_target": 7.0,
        "target_oracle": {"x": 1234},
    }
    public = public_transition(result)

    assert public["executed_action"] == {"role": "UGV", "action": "FORWARD_1"}
    assert public["effect_at"] == 10.0
    assert public["ready_at"] == 16.0
    assert not ({"gt_visible", "dist_to_target", "target_oracle"} & public.keys())
    assert result["gt_visible"] is True


def test_observation_strips_dynamic_gt_without_losing_public_event_fields():
    from agos.methods.cooperative_search.contracts import public_observation

    observation = LiteObservation(
        "public-boundary-ep", 3, 80, "UGV",
        target_reference_image=b"reference-fixture",
        available_actions=["FORWARD_1", "STOP_OBSERVE", "CONFIRM"],
        target_reference_views={"FRONT": b"front-reference-fixture"},
    )
    observation.current_fine_node_id = "ground-node-7"
    observation.internal_vlm_calls_remaining = 12
    observation.coop_event_time = 16.0
    observation.gt_visible = True
    observation.dist_to_target = 7.0
    observation.target_oracle = {"x": 1234}

    public = public_observation(observation)

    assert isinstance(public, LiteObservation)
    assert public is not observation
    assert public.current_fine_node_id == "ground-node-7"
    assert public.internal_vlm_calls_remaining == 12
    assert public.coop_event_time == 16.0
    assert public.target_reference_image == b"reference-fixture"
    assert public.target_reference_views == {"FRONT": b"front-reference-fixture"}
    assert public.available_actions == ["FORWARD_1", "STOP_OBSERVE", "CONFIRM"]
    declared = {field.name for field in fields(LiteObservation)}
    assert not ({"gt_visible", "dist_to_target", "target_oracle"} & vars(public).keys())
    assert set(vars(public)) - declared == {
        "current_fine_node_id", "internal_vlm_calls_remaining", "coop_event_time"
    }
    public.available_actions.append("REJECT")
    assert observation.available_actions == ["FORWARD_1", "STOP_OBSERVE", "CONFIRM"]


def test_config_rejects_unknown_arm():
    from agos.methods.cooperative_search.contracts import validate_config

    with pytest.raises(ValueError, match="arm"):
        validate_config({"arm": "unregistered_arm", "revision": "coop-event-v10"})

def test_arms_form_a_two_by_two_over_two_factors():
    from agos.methods.cooperative_search.contracts import (
        ARMS, EVENT_WAKE_ARMS, JOINT_COVERAGE_ARMS)

    assert ARMS == {"baseline_a", "agos_a", "agos_b", "agos_c"}
    assert JOINT_COVERAGE_ARMS == {"agos_a", "agos_b"}
    assert EVENT_WAKE_ARMS == {"agos_b", "agos_c"}
    # All four (coverage, wake) cells are occupied exactly once.
    cells = {(arm in JOINT_COVERAGE_ARMS, arm in EVENT_WAKE_ARMS) for arm in ARMS}
    assert cells == {(False, False), (True, False), (True, True), (False, True)}
