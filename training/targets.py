"""Map a recorded expert decision to the deployed action vocabulary."""
CONTROL_FIELDS = ("owner_choice", "request_reply", "request_id", "request_assistance",
                  "sender_policy", "resume_search")
ACTION_FIELDS = ("role", "action", "waypoint_id", "candidate_id", "target_verdict",
                 "evidence_view", "proposal_id", "fallback_action", "fallback_waypoint_id",
                 "geometry_type", "position_xy", "localization_source", "source_frame_id",
                 "source_image_sha256", "selected_proposal")
UAV_EXTRA = frozenset({"SUSPECT_CANDIDATE"})


def canonical_target(target):
    """Flatten one archived expert/model target into the deployable JSON object."""
    target = target or {}
    control = {key: target[key] for key in CONTROL_FIELDS
               if target.get(key) not in (None, "", False)}
    inner = target.get("action")
    if isinstance(inner, str):
        inner = target
    if not isinstance(inner, dict):
        # A crop-verification label is flat (``selected_proposal`` + verdict) and
        # carries no ``action`` object; keep it instead of dropping the row.
        inner = target if any(key in target for key in ACTION_FIELDS) else {}
    flat = {key: inner[key] for key in ACTION_FIELDS
            if inner.get(key) not in (None, "", [], {})}
    flat.update(control)
    return flat


SURFACE_ACTION = {"ANNOTATE_CANDIDATE": "SUSPECT_CANDIDATE"}


def surface_target(row, target):
    """Map an executed expert action onto the model-visible action surface.

    The UAV cannot emit ANNOTATE_CANDIDATE directly: it selects a proposal with
    SUSPECT_CANDIDATE and the session's verifier performs the annotation.  A
    row that trained the raw expert action would teach an illegal action.
    """
    action = target.get("action")
    mapped = SURFACE_ACTION.get(action)
    if not mapped:
        return target, None
    support = (row.get("expert_decision") or {}).get("observable_support") or {}
    proposal_id = support.get("proposal_id")
    if not proposal_id:
        return {}, "expert annotation has no public proposal id"
    available = {str(item.get("proposal_id")) for item in
                 (row.get("public_input") or {}).get("proposals") or []
                 if isinstance(item, dict)}
    if available and str(proposal_id) not in available:
        return {}, "expert proposal id absent from the recorded public menu"
    return ({"role": target.get("role") or "UAV", "action": mapped,
             "proposal_id": proposal_id}, None)


def _legal(public_input, target):
    action = target.get("action")
    if not isinstance(action, str) or not action:
        return True
    role = public_input.get("role") or target.get("role")
    menu = set(((public_input.get("legal_menu") or {}).get("actions") or []))
    if role == "UAV":
        menu |= UAV_EXTRA
    return action in menu
