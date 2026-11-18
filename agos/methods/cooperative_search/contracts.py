"""Deployment-only input contracts, separate from the privileged training expert."""
from copy import deepcopy
from dataclasses import fields

from agos.lite_core import CandidateInfo, LiteAction, LiteObservation

MODES = frozenset({"cooperative_search"})
#: 2x2 design on top of the shared event skeleton: coverage (fixed|joint) and
#: wake rhythm (every observation|only events). ``agos_c`` completes the square
#: so each factor can be read at both levels of the other; without it the two
#: factors are confounded and only the two one-factor contrasts are legible.
ARMS = frozenset({"baseline_a", "agos_a", "agos_b", "agos_c"})
JOINT_COVERAGE_ARMS = frozenset({"agos_a", "agos_b"})
EVENT_WAKE_ARMS = frozenset({"agos_b", "agos_c"})
REVISION = "coop-event-v10"
#: Every cooperative_search key any module reads. A config key outside this
#: set is a silent no-op (2026-10-28 audit: ``routine_fraction`` shipped in all
#: three arm yamls while the controller read ``routine_progress``; the values
#: matched only by coincidence), so unknown keys are a config error.
ALLOWED_CONFIG_KEYS = frozenset({
    "revision", "arm", "expert", "capture_root",
    "verify_system_v2",
    "footprint_radius_m", "uav_speed_mps", "ugv_speed_mps",
    "intersection_max_distance_m", "uav_fov_deg", "grid_spacing_m",
    "routine_per_role", "routine_progress",
    "wake_identity_min", "evidence_viewpoint_m",
    "gsam_endpoint", "ugv_endpoint", "tool_timeout_s",
    "box_threshold", "text_threshold", "ugv_view_fov_deg",
})


def _pick(value, names):
    return {key: deepcopy(value[key]) for key in names if key in value}


def public_context(context: dict) -> dict:
    plan = context.get("episode_plan") or context
    out = _pick(context, ("episode_id", "target_reference_image", "target_reference_views"))
    region = _pick(plan.get("region") or {}, ("x_min", "x_max", "y_min", "y_max", "size_m"))
    out["region"] = region
    out["start"] = {
        role: _pick(pose, ("x", "y", "z", "yaw"))
        for role, pose in (plan.get("start") or {}).items()
        if role in ("uav", "ugv")
    }
    out["target_description"] = str((plan.get("target_task") or {}).get("target_description", ""))
    grid = []
    for wp in plan.get("uav_grid_waypoints") or []:
        if not wp.get("active", True):
            continue
        x, y = wp["x"], wp["y"]
        if not (region["x_min"] <= x <= region["x_max"] and
                region["y_min"] <= y <= region["y_max"]):
            continue
        grid.append(_pick(wp, ("id", "x", "y", "z", "active", "visited")))
    out["uav_grid_waypoints"] = grid
    return out


def _candidate(value):
    if value is None:
        return None
    return CandidateInfo(**{
        field.name: deepcopy(getattr(value, field.name)) for field in fields(CandidateInfo)
    })


def public_observation(obs: LiteObservation) -> LiteObservation:
    data = {field.name: deepcopy(getattr(obs, field.name))
            for field in fields(LiteObservation)
            if field.name not in ("candidates", "active_candidate")}
    data["candidates"] = [_candidate(item) for item in obs.candidates]
    data["active_candidate"] = _candidate(obs.active_candidate)
    clean = LiteObservation(**data)
    for key in ("raw_agent_image", "uav_frame", "uav_memory_text", "ugv_memory_text",
                "current_fine_node_id", "internal_vlm_calls_remaining", "coop_event_time",
                "r18_ugv_yaw_deg", "r18_camera_height_m", "observation_contract",
                "candidate_use_ground_elevation"):
        if hasattr(obs, key):
            setattr(clean, key, deepcopy(getattr(obs, key)))
    # These trusted public services operate on the live ledger/public road heights.
    for key in ("candidate_publication_check", "public_ground_elevation"):
        if hasattr(obs, key):
            setattr(clean, key, getattr(obs, key))
    return clean


def public_transition(result: dict) -> dict:
    clean = _pick(result, (
        "step", "action", "candidate_id", "grid_cell", "waypoint_id",
        "candidate_center_xy", "pos", "executed", "error", "status",
        "effect_at", "ready_at", "event_time", "role", "sampled", "frame_id",
        "distance_m", "duration_s",
    ))
    action_fields = tuple(field.name for field in fields(LiteAction))
    for key in ("proposed_action", "executed_action"):
        if key in result:
            value = result[key]
            if isinstance(value, LiteAction):
                value = vars(value)
            if isinstance(value, dict):
                clean[key] = _pick(value, action_fields)
    return clean


def validate_config(config: dict) -> dict:
    cfg = config.get("cooperative_search", config)
    if cfg.get("arm") not in ARMS:
        raise ValueError("cooperative_search: unknown arm (expected one of "
                         + ",".join(sorted(ARMS)) + ")")
    if cfg.get("revision") != REVISION:
        raise ValueError(f'cooperative_search: revision drift, expected {REVISION}')
    unknown = sorted(set(cfg) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError("cooperative_search: unknown config keys (typo or retired key?): "
                         + ",".join(unknown))
    for key in ("diagnostic", "diagnostics"):
        diag = config.get(key) or {}
        if diag.get("oracle_candidate_injection") or diag.get("shield", "none") != "none":
            raise ValueError("cooperative_search: privileged deployment diagnostic forbidden")
    return deepcopy(cfg)
