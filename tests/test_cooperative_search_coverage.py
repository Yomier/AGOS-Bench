"""Executable directed coverage and causal observation progress specifications."""
from types import SimpleNamespace as NS

import pytest

from agos.lite_core import LiteAction, LiteObservation
from agos.methods.base import MethodServices


def fixture():
    coordinates = {"a": (0, 0), "b": (40, 0), "c": (80, 0), "island": (120, 0)}
    edges = {"a": [("FORWARD_1", "b", 40)],
             "b": [("FORWARD_1", "c", 40)], "c": [], "island": []}

    class Nav:
        port_owner = {key: key for key in coordinates}

        def get_maneuver_options(self, port, **kwargs):
            return {action: NS(target_fine_node_id=target, path_distance_m=distance)
                    for action, target, distance in edges[port]}

    services = MethodServices(
        road_graph=NS(nodes={key: NS(x=xy[0], y=xy[1]) for key, xy in coordinates.items()}),
        semantic_nav=Nav(),
    )
    public = {
        "region": {"x_min": -1, "x_max": 121, "y_min": -1, "y_max": 1},
        "start": {"uav": {"x": 0, "y": 0, "z": 80},
                  "ugv": {"x": 0, "y": 0, "z": 0.5}},
        "uav_grid_waypoints": [{"id": f"U{n}", "x": n * 40, "y": 0}
                               for n in range(4)],
    }
    return services, public


def planner(joint=False):
    from agos.methods.cooperative_search.coverage import CoveragePlanner
    services, public = fixture()
    result = CoveragePlanner(services, public, {"footprint_radius_m": 25, "grid_spacing_m": 40})
    result.start(joint=joint)
    return result


def test_fixed_routes_are_deterministic_and_independent():
    first, second = planner(), planner()
    assert first.summary("UAV", 0)["station_order"] == second.summary("UAV", 0)["station_order"]
    assert first.summary("UGV", 0)["assigned_units"] == ["a", "b", "c"]
    assert first.summary("UAV", 0)["assigned_units"] == ["a", "b", "c", "island"]


def test_joint_allocation_removes_avoidable_duplicate_work():
    coverage = planner(joint=True)
    air = set(coverage.summary("UAV", 0)["assigned_units"])
    ground = set(coverage.summary("UGV", 0)["assigned_units"])
    assert not air.intersection(ground)
    assert air | ground == {"a", "b", "c", "island"}


def test_unreachable_ground_unit_assigned_to_air():
    coverage = planner(joint=True)
    assert "island" in coverage.summary("UAV", 0)["assigned_units"]
    assert "island" not in coverage.summary("UGV", 0)["assigned_units"]


def test_unreachable_by_both_is_reported():
    from agos.methods.cooperative_search.coverage import CoveragePlanner
    services, public = fixture()
    public["uav_grid_waypoints"] = public["uav_grid_waypoints"][:2]
    coverage = CoveragePlanner(services, public, {"footprint_radius_m": 25})
    coverage.start(joint=True)
    assert coverage.summary("UAV", 0)["unreachable_units"] == ["island"]


def test_failed_move_does_not_advance_and_sampling_is_causal():
    coverage = planner()
    obs = LiteObservation("ep", 1, 80, "UAV", agent_position=(0, 0, 80))
    action = LiteAction("UAV", "MOVE_TO_UAV_WAYPOINT", waypoint_id="U1")
    coverage.observe(obs, action, {"executed": False, "sampled": True,
                                  "pos": (40, 0, 80), "ready_at": 10})
    assert coverage.summary("UAV", 20)["completed_units"] == []
    coverage.observe(obs, action, {"executed": True, "sampled": True,
                                  "pos": (40, 0, 80), "ready_at": 10})
    assert coverage.summary("UAV", 0)["completed_units"] == []
    assert coverage.summary("UAV", 10)["completed_units"] == ["b"]
    assert coverage.summary("UAV", 0)["completed_units"] == []


def test_reallocate_preserves_completed_units():
    coverage = planner(joint=True)
    obs = LiteObservation("ep", 1, 80, "UAV", agent_position=(120, 0, 80))
    coverage.observe(obs, LiteAction("UAV", "WAIT"),
                     {"executed": True, "sampled": True,
                      "pos": (120, 0, 80), "ready_at": 4})
    coverage.reallocate("verification_end", {"UAV": (120, 0), "UGV": (40, 0)}, 4)
    assert "island" in coverage.summary("UAV", 4)["completed_units"]
    assert "island" not in coverage.summary("UAV", 4)["assigned_units"]


def test_ground_route_does_not_invent_reverse_edges():
    coverage = planner()
    obs = LiteObservation("ep", 1, 80, "UGV", agent_position=(80, 0, 0.5),
                          available_actions=["STOP_OBSERVE"])
    obs.current_fine_node_id = "c"
    obs.coop_event_time = 0
    assert coverage.next_action(obs).action == "STOP_OBSERVE"


def test_uav_global_route_is_expanded_through_current_legal_neighbors():
    coverage = planner()
    obs = LiteObservation("ep", 1, 80, "UAV", agent_position=(0, 0, 80),
                          available_actions=["MOVE_TO_UAV_WAYPOINT", "WAIT"],
                          available_waypoints=[{"waypoint_id": "U1", "world_xy": (40, 0)}])
    obs.coop_event_time = 0
    action = coverage.next_action(obs)
    assert action.action == "MOVE_TO_UAV_WAYPOINT"
    assert action.waypoint_id == "U1"
