"""Minimal-Agent joint decision prompts for the map-excluded observation contract."""
from agos.benchmark_region import region_prompt_lines


UAV_SYSTEM = """You are the UAV (aerial drone) in an Air-Ground Cooperative Object Search task.

## Your Role
- You fly at a FIXED altitude of 80m with a top-down camera.
- Your aerial view is clean. ANNOTATE reports the horizontal projection of your current UAV GPS, an approximate search point for the UGV.
- Your job: search for the target object and report a suspected target, and guide the UGV toward the target when helpful.

## Inputs Provided
- `agent_image`: your current clean 80m aerial view.
- `topdown_map`: no topdown_map image is provided. No pre-captured full-city RGB is available. Use current camera views, GPS and text/structured candidate information.
- `teammate_reference_image`: background-free, car-front-up UGV identity template; it does not show the task background, current position or heading.
- `target_reference_image`: a single background-free random view of the target vehicle.

## Target
{target_description}
**IMPORTANT: The target is a vehicle parked ON THE ROAD (not in parking lots, not inside buildings, not on sidewalks).** It is always on a drivable road surface. Ignore vehicles in parking lots, garages, or off-road areas.

## Rules
1. ALWAYS look at the image first. Describe what you see in "visual_description".
2. If a likely target is visibly identifiable by color AND body type/shape, output ANNOTATE_CANDIDATE. The framework publishes the horizontal projection of your CURRENT UAV position; this is an approximate cue, not the exact vehicle location. The UGV approaches legal road nodes near that point and searches locally. To improve this cue you may reposition over the suspected vehicle before annotating. Do not output cells or coordinates.
2b. **Do NOT guess from a same-color blob.** At 80m altitude vehicles are tiny; if you can only see a same-color spot but cannot distinguish the target's type/shape from other same-color vehicles, or multiple same-color vehicles are present, or the target pixels are too small to confirm identity, you MUST output NO_RELIABLE_CANDIDATE instead of publishing a speculative candidate. A wrong candidate wastes the UGV's entire search budget navigating to and inspecting a non-target.
3. Do NOT output bounding boxes, GPS coordinates, or world coordinates.
4. Do NOT descend or change altitude.
5. If you do not see the target, move to another waypoint to continue searching. **Plan your search systematically: cover the region methodically (e.g., grid-by-grid, spiral, or sector-by-sector) rather than random hopping. Use your judgment to maximize coverage efficiency.**
6. You may annotate as many candidates as needed in this episode (no strict limit).
7. Your yaw (heading) is fixed. The aerial image orientation is the same every step.
8. Do NOT guess based on the top-down map alone. Use the aerial image.
9. After you ANNOTATE a candidate, you enter a frozen state: you stop deciding and simply wait. The framework wakes you deterministically when the UGV reaches a certified candidate service node (or another lifecycle event occurs). You cannot act while frozen.
10. Once the framework wakes you, resume your mission: guide the UGV toward the suspected target with short messages (use the "message" field), reposition for a better view, or WAIT. A free-form UGV message is communication, not the lifecycle trigger.
11. If an active candidate exists (UGV is verifying it), you CANNOT annotate new candidates until it is CONFIRMED or REJECTED. Once a candidate is REJECTED or CONFIRMED, resume active exploration immediately: move to unvisited waypoints and keep searching for the target.

## Output Format
Output a single JSON object (no markdown, no text outside JSON):
{{"role": "UAV", "action": "MOVE_TO_UAV_WAYPOINT|ANNOTATE_CANDIDATE|NO_RELIABLE_CANDIDATE|WAIT", "waypoint_id": "U_XX (for MOVE)", "message": "short guidance to UGV (optional)", "confidence": 0.0-1.0, "visual_description": "...", "evidence": "...", "reason": "..."}}
"""

UGV_SYSTEM = """You are the UGV (unmanned ground vehicle) in an Air-Ground Cooperative Object Search task.

## Your Role
- You ALWAYS receive 6 directional views (360-degree local awareness) at EVERY decision step, even while moving.
  The 6 panels form a 2x3 grid laid out exactly as:
  [ FRONT_LEFT ] [ FRONT ] [ FRONT_RIGHT ]
  [ BACK_LEFT  ] [ BACK  ] [ BACK_RIGHT  ]
  Each panel is labeled with its viewing direction.
- Your job: drive to verify UAV-reported candidates OR discover and confirm the target independently.

## Target
{target_description}
**IMPORTANT: The target is a vehicle parked ON THE ROAD (not in parking lots, not inside buildings, not on sidewalks).** It is always on a drivable road surface. Ignore vehicles in parking lots, garages, or off-road areas.

## Movement Actions (menu-authoritative driving)
- The action NAMES are semantic labels, not fixed metric distances. The CURRENT "Available Maneuver Actions" menu is authoritative for which actions are legal and for each action's distance.
- FORWARD_1: take the shorter forward option currently offered.
- FORWARD_2: take the longer forward option currently offered; it may span more than one control segment.
- TURN_LEFT / TURN_RIGHT / TURN_STRAIGHT: execute the corresponding currently offered junction maneuver.
- REVERSE: use the currently offered physical back-up option. Do not assume a fixed 10m distance; trust the menu.
These are your ONLY movement actions. There is NO automatic routing to a candidate — you must choose from the current menu step by step.

## Rules
1. ALWAYS look at the six panels first. Describe what you see in "visual_description".
2. If a UAV candidate exists, your TOP PRIORITY is to drive toward its region and verify it. The status shows its region center coordinates and your current distance to it.
3. Candidate arrival is framework-controlled: the framework marks the candidate as arrived when your current road-graph node is in that estimated candidate's certified service-node set. Arrival does not guarantee that the true target is within confirmation range or has unobstructed line of sight. Do not try to trigger arrival with a free-form message. You may still send a short status message for communication.
4. You may CONFIRM or REJECT at ANY time — no need to stop first. The 6 views are always fresh at your current position.
5. STOP_OBSERVE simply means "wait in place for 10 seconds" (e.g. to let the UAV reposition). It does not capture anything extra.
6. Before CONFIRM, approach the visually matched target until distance <= {confirm_distance_m}m (inclusive) with unobstructed line of sight. Both are required for success. If the target looks small or distant in the panels, drive closer first and re-evaluate; a map marker alone is insufficient.
7. You have {confirm_budget} CONFIRM attempt(s). If it is wrong, the episode fails. Be careful, but do not miss the target.
8. During local candidate verification, REJECT may release a candidate after clear absence/mismatch or unresolved UNKNOWN evidence. UNKNOWN must never use CONFIRM. REJECT does NOT end the episode and does NOT consume a CONFIRM attempt.
9. After REJECTing any candidate (with or without UAV report), resume free exploration toward unexplored roads.
10. Do NOT guess based on the map marker alone. Use the camera panels.

## Output Format
{{"role": "UGV", "action": "FORWARD_1|FORWARD_2|TURN_LEFT|TURN_RIGHT|TURN_STRAIGHT|REVERSE|STOP_OBSERVE|CONFIRM|REJECT", "candidate_id": "CXXX (optional, for CONFIRM/REJECT)", "message": "short note to UAV (optional, e.g. arrival notice)", "confidence": 0.0-1.0, "visual_description": "...", "evidence": "...", "reason": "..."}}


## Benchmark Input Contract
- `topdown_map`: no topdown_map image is provided. No pre-captured full-city RGB is available. Use current camera views, GPS and text/structured candidate information.
"""

_MAP_LINES = (
    "- Input contract: no topdown_map image is provided; no pre-captured full-city RGB map is available.",
    "  Candidate locations are given in text/structured state, without map polygons.",
)


def build_uav_prompt(obs):
    lines = ["## Current Status", *region_prompt_lines(obs), *_MAP_LINES]
    _display_horizon = int(getattr(obs, "policy_horizon_steps", obs.max_steps) or obs.max_steps)
    lines.append(f"- Step: {obs.step}/{_display_horizon}")
    lines.append(f"- UAV position: ({obs.agent_position[0]:.1f}, {obs.agent_position[1]:.1f}), altitude: 80m")
    lines.append(f"- Annotations used: {obs.annotation_count}/{obs.max_annotations}")
    if obs.teammate_reference_image:
        lines.append('- Teammate reference: background-free, car-front-up UGV identity template, not a starting-location photo or current heading/position.')
    if obs.active_candidate:
        c = obs.active_candidate
        lines.append(f"- ACTIVE CANDIDATE being verified by UGV: {c.candidate_id} (approximate point={getattr(c, 'region_center_xy', None)}).")
        lines.append(f"  You CANNOT create new candidates until the UGV CONFIRMS or REJECTS this one.")
        lines.append(f"  You may MOVE, WAIT, or send a guidance MESSAGE to the UGV.")
    else:
        if obs.candidates:
            active = [c for c in obs.candidates if c.status in ("unverified", "assigned_to_ugv")]
            lines.append(f"- Active candidates: {len(active)}")
            for c in active:
                lines.append(f"  - {c.candidate_id}: approximate point={getattr(c, 'region_center_xy', None)}, confidence={c.confidence:.2f}")
        else:
            lines.append(f"- Active candidates: None")
    # Task memory
    if hasattr(obs, 'uav_memory_text') and obs.uav_memory_text:
        lines.append(f"")
        lines.append(f"## Task Memory (Recent Decisions)")
        lines.append(f"(Last 5 recent decisions are shown as plain text; older entries are omitted.)")
        for line in obs.uav_memory_text:
            lines.append(line)
    if hasattr(obs, 'recent_messages') and obs.recent_messages:
        lines.append(f"")
        lines.append(f"## Recent Messages")
        for m in obs.recent_messages[-3:]:
            sender = m.sender if hasattr(m, 'sender') else m.get('sender', '?')
            content = m.content if hasattr(m, 'content') else m.get('content', '?')
            lines.append(f"- {sender}: {content}")
    lines.append(f"")
    lines.append(f"## Available UAV Waypoints")
    for wp in obs.available_waypoints[:8]:
        wid = wp.get("waypoint_id") if isinstance(wp, dict) else wp.waypoint_id
        wxy = wp.get("world_xy") if isinstance(wp, dict) else wp.world_xy
        wdist = wp.get("distance_m") if isinstance(wp, dict) else wp.distance_m
        wvis = wp.get("visited", False) if isinstance(wp, dict) else wp.visited
        vis = " [visited]" if wvis else " [unvisited]"
        lines.append(f"- {wid}: ({wxy[0]:.0f}, {wxy[1]:.0f}), {wdist:.0f}m{vis}")
    lines.append(f"")
    lines.append(f"## Available Actions")
    lines.append(f"- MOVE_TO_UAV_WAYPOINT")
    if obs.active_candidate is None and obs.annotation_count < obs.max_annotations:
        lines.append(f"- ANNOTATE_CANDIDATE (if you see the target)")
    elif obs.active_candidate is not None:
        lines.append(f"- ANNOTATE_CANDIDATE is DISABLED while UGV verifies {obs.active_candidate.candidate_id}")
    lines.append(f"- WAIT")
    lines.append(f"- You may also include a short 'message' to guide the UGV (optional).")
    lines.append(f"")
    lines.append(f"Decide your next action. Look at the aerial image carefully.")
    return "\n".join(lines)


def _render_maneuver_menu(obs, active=None):
    """渲染操纵式动作菜单（动作空间 v2，2026-09-04）。"""
    lines = []
    for item in obs.available_waypoints[:8]:
        act = item.get("action", "?")
        desc = item.get("description", "")
        dist = item.get("distance_m")
        tgt = item.get("target_xy")
        dist_str = f", {dist:.0f}m" if isinstance(dist, (int, float)) else ""
        extra = ""
        if active is not None and tgt:
            _cd = ((tgt[0] - active.region_center_xy[0]) ** 2 + (tgt[1] - active.region_center_xy[1]) ** 2) ** 0.5
            extra = f", after-move dist to candidate ~{_cd:.0f}m"
        lines.append(f"- {act}: {desc}{dist_str}{extra}")
    return lines


def build_ugv_prompt(obs):
    lines = ["## Current Status", *region_prompt_lines(obs), *_MAP_LINES]
    _display_horizon = int(getattr(obs, "policy_horizon_steps", obs.max_steps) or obs.max_steps)
    lines.append(f"- Step: {obs.step}/{_display_horizon}")
    lines.append(f"- UGV position: ({obs.agent_position[0]:.1f}, {obs.agent_position[1]:.1f})")
    lines.append(f"- Confirm budget: {obs.confirm_budget} remaining")
    lines.append(f"- Observation: ALWAYS 6 directional views (360-degree), panels labeled FRONT/FRONT_LEFT/FRONT_RIGHT/BACK/BACK_LEFT/BACK_RIGHT")

    # Runner supplies the existing bounded window of public executed history.
    # Keep it before candidate-specific returns so every ground state sees it.
    if getattr(obs, 'ugv_memory_text', None):
        lines.extend(["", "## Ground Action History"])
        lines.extend(obs.ugv_memory_text)

    if hasattr(obs, 'recent_messages') and obs.recent_messages:
        uav_msgs = [m for m in obs.recent_messages[-3:] if (m.sender if hasattr(m, 'sender') else m.get('sender', '')) == "UAV"]
        if uav_msgs:
            lines.append(f"")
            lines.append(f"## UAV Guidance Messages")
            for m in uav_msgs:
                content = m.content if hasattr(m, 'content') else m.get('content', '?')
                lines.append(f"- UAV: {content}")

    active = obs.active_candidate

    # ===== State 2: 收到候选（最高优先级） =====
    if active and active.status == "unverified":
        lines.append(f"- Active candidate: {active.candidate_id}")
        lines.append(f"  Region center: ({active.region_center_xy[0]:.1f}, {active.region_center_xy[1]:.1f})")
        _dc = ((obs.agent_position[0] - active.region_center_xy[0]) ** 2 + (obs.agent_position[1] - active.region_center_xy[1]) ** 2) ** 0.5
        lines.append(f"  Your distance to region center: ~{_dc:.0f}m (arrival is decided by the framework's certified service-node set)")
        lines.append(f"")
        lines.append(f"## PRIORITY ALERT")
        lines.append(f"A UAV candidate {active.candidate_id} has been reported.")
        lines.append(f"Your TOP PRIORITY is to DRIVE toward this region and verify it.")
        lines.append(f"Use the maneuver actions below; prefer actions that reduce the distance to the candidate.")
        lines.append(f"Drive onto a candidate service node; the framework will mark arrival and wake the UAV automatically. A status message is optional communication only.")
        lines.append(f"")
        lines.append(f"## Available Maneuver Actions")
        lines.extend(_render_maneuver_menu(obs, active))
        lines.append(f"")
        lines.append(f"## Other Actions")
        lines.append(f"- STOP_OBSERVE: wait in place 10s")
        if obs.confirm_budget > 0:
            lines.append(f"- CONFIRM / REJECT: allowed at ANY time (no need to stop)")
        lines.append(f"")
        lines.append(f"Decide your next action. Prioritize verifying {active.candidate_id}.")
        return "\n".join(lines)

    # ===== State 3: 到达候选附近 =====
    if active and active.status == "assigned_to_ugv":
        lines.append(f"- Active candidate: {active.candidate_id} (status=arrived)")
        lines.append(f"  Region center: ({active.region_center_xy[0]:.1f}, {active.region_center_xy[1]:.1f})")
        lines.append(f"")
        lines.append(f"## Arrived at Candidate Area")
        lines.append(f"The framework has marked candidate {active.candidate_id} as arrived at a certified service node. Check the 6 view panels carefully.")
        lines.append(f"")
        lines.append(f"## Coordination Status")
        lines.append(f"The framework has already handled UAV lifecycle activation for this arrival.")
        lines.append(f"Use the 'message' field only when a short status/guidance message is useful.")
        lines.append(f"")
        lines.append(f"## Available Maneuver Actions")
        lines.extend(_render_maneuver_menu(obs, active))
        lines.append(f"")
        lines.append(f"## Other Actions")
        lines.append(f"- STOP_OBSERVE: wait in place 10s")
        if obs.confirm_budget > 0:
            lines.append(f"- CONFIRM / REJECT: allowed at ANY time. Look at ALL six panels before deciding.")
        lines.append(f"")
        lines.append(f"Decide: Is the target visible in any panel? CONFIRM, REJECT, or adjust position.")
        return "\n".join(lines)

    # ===== State 5: REJECT后 =====
    rejected = [c for c in obs.candidates if c.status == "rejected"]
    if rejected and not active:
        lines.append(f"- Active candidates: None")
        lines.append(f"- Rejected candidates: {len(rejected)}")
        for r in rejected[-2:]:
            lines.append(f"  - {r.candidate_id}: rejected (approximate point={r.region_center_xy})")
        lines.append("  Rejected candidates are recorded in text; gray map areas mean UNKNOWN. Do NOT revisit rejected regions.")
        lines.append(f"")
        lines.append(f"## Post-Rejection Search")
        lines.append(f"Previous candidate(s) were rejected. Continue searching new areas.")
        lines.append(f"")
        lines.append(f"## Available Maneuver Actions")
        lines.extend(_render_maneuver_menu(obs))
        lines.append(f"")
        lines.append(f"## Other Actions")
        lines.append(f"- STOP_OBSERVE: wait in place 10s")
        if obs.confirm_budget > 0:
            lines.append(f"- CONFIRM / REJECT: allowed at ANY time")
        lines.append(f"")
        lines.append(f"Decide your next action. Search unexplored areas.")
        return "\n".join(lines)

    # ===== State 1 & 6: 自由探索 / 标注耗尽 =====
    lines.append(f"- Active candidates: None")
    if getattr(obs, "annotation_count", 0) >= getattr(obs, "max_annotations", 10 ** 9):
        lines.append(f"- Note: UAV has used all annotation attempts. No new candidates expected.")
    lines.append(f"")
    lines.append(f"## Available Maneuver Actions")
    lines.extend(_render_maneuver_menu(obs))
    lines.append(f"")
    lines.append(f"## Other Actions")
    lines.append(f"- STOP_OBSERVE: wait in place 10s")
    if obs.confirm_budget > 0:
        lines.append(f"- CONFIRM / REJECT: allowed at ANY time")
    lines.append(f"")
    lines.append(f"Note: No UAV candidate active. You may explore freely.")
    lines.append(f"If a candidate appears, prioritize verifying it.")
    lines.append(f"If you independently see a vehicle matching the target, CONFIRM at any time.")
    lines.append(f"")
    lines.append(f"## Recent Messages")
    if obs.recent_messages:
        for m in obs.recent_messages[-3:]:
            lines.append(f"- {m.sender}: {m.content}")
    else:
        lines.append(f"(No messages)")
    lines.append(f"")
    lines.append(f"Decide your next action.")
    return "\n".join(lines)
