"""Public phase/target contracts for cooperative-search event capture."""
from __future__ import annotations

PHASES = frozenset({"coverage", "receive", "approach", "arrival", "verify", "recover"})
CALL_REASONS = frozenset({"new_evidence", "request_receipt", "arrival", "verify",
                          "fault", "routine_check", "observation"})
FORBIDDEN_INPUT_KEYS = frozenset({
    "gt", "gt_visible", "target_oracle", "dist_to_target", "distractor_oracle",
    "oracle", "solvability", "target_xy", "reasoning", "chain_of_thought", "cot",
})
TARGET_ACTION_KEYS = frozenset({
    "role", "action", "waypoint_id", "candidate_id", "target_verdict", "evidence_view",
    "proposal_id", "fallback_action", "fallback_waypoint_id", "geometry_type",
    "position_xy", "localization_source", "source_frame_id", "source_image_sha256",
    "owner_choice", "request_reply", "request_id", "request_assistance", "sender_policy",
    "resume_search",
    "selected_proposal",
})


def public_phase(state: dict, public_events: list, event_time: float) -> dict:
    """Phase comes from public causal state, never from a predicted label."""
    phase = (state or {}).get("phase", "coverage")
    if phase not in PHASES:
        raise ValueError("unknown public phase: " + str(phase))
    reason = (state or {}).get("call_reason")
    if reason is not None and reason not in CALL_REASONS:
        raise ValueError("unknown public call reason: " + str(reason))
    for event in public_events or []:
        if float(event.get("available_at", 0)) > float(event_time):
            raise ValueError("public event used before its visible time")
    return {"task_phase": phase, "call_reason": reason, "event_time": float(event_time)}


def student_input(public_input: dict, public_state: dict) -> dict:
    """Filtered deployment-side input: no GT, no future facts, no CoT."""
    merged = dict(public_input or {})
    merged.update({key: value for key, value in (public_state or {}).items()
                   if key in ("task_phase", "call_reason", "event_time")})
    for key in merged:
        if key.lower() in FORBIDDEN_INPUT_KEYS:
            raise ValueError("privileged key in student input: " + key)
    return merged


def validate_target(public_input: dict, target: dict) -> list:
    """Return explicit target problems; an empty list means trainable."""
    problems = []
    if not isinstance(target, dict) or not target:
        return ["missing target"]
    unknown = set(target) - TARGET_ACTION_KEYS
    if unknown:
        problems.append("unknown target keys: " + ",".join(sorted(unknown)))
    role = public_input.get("role")
    if target.get("role") and role and target["role"] != role:
        problems.append("target role does not match input role")
    action = target.get("action")
    if isinstance(action, dict):
        action = action.get("action")
    if action is not None and not isinstance(action, str):
        problems.append("action must be a string")
    if isinstance(target.get("position_xy"), (list, tuple)):
        if len(target["position_xy"]) != 2:
            problems.append("position_xy must have two numbers")
    if target.get("target_verdict") not in (None, "MATCH", "NON_MATCH", "UNKNOWN"):
        problems.append("invalid target_verdict")
    return problems
