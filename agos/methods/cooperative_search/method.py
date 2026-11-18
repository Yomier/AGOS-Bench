"""Cooperative visual-search method session."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, fields

from agos.lite_core import LiteAction
from agos.methods.common.grounded_localizer import project_observation_proposal
from agos.methods.common.route_tool import SparseRouteTool

from .contracts import (JOINT_COVERAGE_ARMS, REVISION, public_context, public_observation,
                        public_transition, validate_config)
from .controller import CooperativeController
from .coverage import CoveragePlanner
from .perception import ProposalScreen
from .policy import DecisionPolicy

NAMES = {"baseline_a": "Baseline-Agent", "agos_a": "AGOS-Agent-A",
         "agos_b": "AGOS-Agent-B", "agos_c": "AGOS-Agent-C"}


def _short_id(value):
    """Proposal ids travel as ``P1`` on the board and ``<sha>/V2/P1`` in evidence.

    Both shapes address the same proposal, so equality is decided on the last
    path segment; without this the model's ``<sha>/V2/P1`` never matched the
    board's ``P1`` and the nomination died as ``proposal_not_in_menu``.
    """
    return str(value or "").rsplit("/", 1)[-1]


#: 2026-10-31 v7: the UGV covers ~14 m of road per team step, so a candidate
#: published 300-600 m away cannot be reached inside the 80-step contract.
#: v6 evidence (lite_easy_00066): the UGV drove 1,138 m, closed from 603 m to
#: 114 m and ran out of steps without ever claiming -- the nomination was
#: simply out of range. Nominations therefore require the candidate to be
#: within this straight-line distance of the UGV's current position.
PUBLISH_MAX_UGV_DISTANCE_M = 250.0


class _ProposalView:
    """Read-only geometry the localizer projector expects, built from a board dict."""

    __slots__ = ("bbox_xyxy", "mask_bbox_xyxy", "proposal_id")

    def __init__(self, item: dict):
        self.bbox_xyxy = item.get("bbox_xyxy")
        self.mask_bbox_xyxy = item.get("mask_bbox_xyxy")
        self.proposal_id = item.get("proposal_id")


class CooperativeSearchMethod:
    method_id = "cooperative_search"
    public_input_only = True

    def __init__(self, config=None, services=None, profile=None):
        self.config = config or {}
        self.cfg = validate_config(self.config)
        self.display_name = NAMES[self.cfg["arm"]]
        self.services = services
        self.profile = profile
        self.policy = DecisionPolicy(self.config, capture_root=self.cfg.get("capture_root"))
        self.uav = self.ugv = self
        self.sys_template = "cooperative_event_v1"
        self.context = {}
        self.last_trace = {}
        self.last_raw_response = self.last_parse_status = ""
        self.last_action_downgraded = False
        self._step_calls = []
        self._expert_gt = None
        self.expert = None
        self.coverage = None
        self.controller = None
        self.screen = None
        self.route_tool = SparseRouteTool(float(self.cfg.get("intersection_max_distance_m", 50)))
        self._sample_keys = set()
        # 2026-11-01: candidate ids already recorded at the expert confirm site.
        self._confirm_recorded = set()
        # 2026-10-31 v10: the radius the success protocol measures the claim at.
        self.max_confirm_distance_m = float(
            (self.config.get("success") or {}).get("max_confirm_distance_m") or 30.0)

    def start_episode(self, episode_context):
        raw = episode_context or {}
        self._expert_gt = self._private_target(raw)
        self.context = public_context(raw)
        expert_cfg = self.cfg.get("expert") or {}
        self.expert = None
        if expert_cfg.get("enabled"):
            if self._expert_gt is None:
                raise ValueError("cooperative_search: expert mode requires a private target")
            from training.expert import EventExpert
            self.expert = EventExpert(expert_cfg)
        self.controller = CooperativeController(self.cfg["arm"], self.cfg)
        self.coverage = CoveragePlanner(self.services, self.context, self.cfg)
        self.coverage.start(joint=self.cfg["arm"] in JOINT_COVERAGE_ARMS)
        self.screen = ProposalScreen(self.cfg)
        self.policy.drain_calls()
        self.policy.events.clear()
        self.last_trace = {}
        self._sample_keys.clear()
        self._confirm_recorded.clear()

    def close_episode(self):
        self.context = {}
        self.coverage = self.controller = self.screen = None
        self.policy.drain_calls()
        self.policy.events.clear()
        self.last_trace = {}
        self._sample_keys.clear()
        self._confirm_recorded.clear()

    def get_mode_name(self):
        return self.display_name

    def role_capabilities(self, role):
        return {"uses_vlm": False, "bind_uav_frame": True, "continue_uav": True}

    def drain_internal_vlm_calls(self):
        return self.policy.drain_calls()

    def _sample(self, obs):
        if obs.role == "UAV":
            frame = getattr(obs, "uav_frame", None)
            images = [frame.image_bytes] if frame is not None and frame.valid else []
        else:
            images = [view.image_bytes for view in obs.ugv_views if view.valid]
            if len(images) != 6:
                images = []
        if not images or not all(images):
            return
        now = float(getattr(obs, "coop_event_time", 0))
        key = (obs.role, tuple(obs.agent_position), now,
               tuple(hashlib.sha256(image).hexdigest() for image in images))
        if key not in self._sample_keys:
            self._sample_keys.add(key)
            self.coverage.observe(obs, None, {"executed": True, "sampled": True,
                                             "pos": obs.agent_position, "ready_at": now})

    def _ingest_messages(self, obs):
        for message in obs.recent_messages:
            try:
                payload = json.loads(message.content)
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") != "COOP_MESSAGE_V1":
                continue
            sender = getattr(message, "sender", "")
            receiver = getattr(message, "receiver", "") or obs.role
            if payload.get("type") == "verification_complete":
                self.controller.roles[obs.role]["sender_policy"] = "resume"
                continue
            if payload.get("type") != "assistance_request":
                continue
            request = {key: deepcopy(payload[key]) for key in
                       ("request_id", "type", "candidate_id", "region_xy", "source_image_sha256")
                       if key in payload}
            request.update(sender=sender, receiver=receiver,
                           message_id=getattr(message, "message_id", payload.get("request_id")),
                           available_at=float(getattr(message, "timestamp_step", 0)))
            candidate = next((candidate for candidate in obs.candidates
                              if (request.get("candidate_id") == candidate.candidate_id or
                                  (request.get("source_image_sha256") and
                                   request["source_image_sha256"] == candidate.source_image_sha256))), None)
            if candidate:
                request["candidate_id"] = candidate.candidate_id
                request["service_node_ids"] = list(candidate.service_node_ids)
                request["region_xy"] = list(candidate.region_center_xy)
            self.controller.ingest_message(request)
        self._implicit_candidate_request(obs)

    def _implicit_candidate_request(self, obs):
        """A public active candidate is itself a handoff, even without a message.

        The runner may publish a candidate through a path that carries no
        COOP_MESSAGE_V1 (or suppresses the annotation message); the candidate is
        still public state the ground agent may service.  The synthesized request
        keeps the same identity/reply contract and consumes one receipt decision.
        """
        candidate = obs.active_candidate
        if candidate is None:
            return
        known = {request.get("candidate_id") for request in self.controller.requests.values()}
        if candidate.candidate_id in known:
            return
        now = float(getattr(obs, "coop_event_time", 0))
        self.controller.ingest_message({
            "request_id": "candidate:%s" % candidate.candidate_id,
            "message_id": "candidate:%s" % candidate.candidate_id,
            "type": "assistance_request", "sender": "UAV", "receiver": "UGV",
            "candidate_id": candidate.candidate_id,
            "region_xy": list(candidate.region_center_xy),
            "service_node_ids": list(candidate.service_node_ids),
            "source_image_sha256": candidate.source_image_sha256,
            "available_at": now})

    def _approach(self, obs):
        request = self.controller.roles[obs.role]["active_request"]
        if not request:
            return {"status": "inactive", "arrived": False}, None
        if obs.role == "UGV":
            ids = request.get("service_node_ids") or []
            candidate = next((item for item in obs.candidates
                              if item.candidate_id == request.get("candidate_id")), None)
            if candidate is not None:
                ids = candidate.service_node_ids
            route = self.route_tool.plan(self.services.semantic_nav,
                                          getattr(obs, "current_fine_node_id", ""),
                                          {"service_node_ids": ids})
            route["arrived"] = route["status"] == "AT_SERVICE_PORT"
            action = route.get("preferred_action")
            if action and action in obs.available_actions:
                return route, LiteAction("UGV", action, candidate_id=request.get("candidate_id"))
            if not route["arrived"]:
                route["fault_id"] = f"approach:{request['request_id']}:{route['status']}"
            return route, None
        point = request.get("region_xy")
        if point is None:
            return {"status": "missing_public_region", "arrived": False,
                    "fault_id": f"approach:{request['request_id']}:missing_region"}, None
        goal = min(self.coverage.grid, key=lambda key: (math.dist(point, self.coverage.grid[key]), key))
        arrived = math.dist(obs.agent_position[:2], self.coverage.grid[goal]) <= 1.0
        action = None if arrived else self.coverage._air_next(obs, goal)
        status = {"arrived": arrived, "status": "arrived" if arrived else "approach", "goal": goal}
        if not arrived and action is None:
            status["fault_id"] = f"approach:{request['request_id']}:unreachable"
        return status, action

    @staticmethod
    def _rest(obs):
        action = "WAIT" if obs.role == "UAV" else "STOP_OBSERVE"
        if action not in obs.available_actions:
            raise ValueError("cooperative_search: no legal observation action")
        return LiteAction(obs.role, action)

    def _drop(self, fallback, reason):
        """Return the fallback action with a public reason the episode log shows.

        A candidate the method refuses to publish must stay diagnosable: without
        this the 2026-10-28 pilot could not tell "model never asked" from
        "method discarded every request".
        """
        self.last_trace["arbitration_reason"] = reason
        fallback.reason = reason
        return fallback

    def _drop_reject(self, obs, target, fallback, reason):
        """Refuse a claim the ground evidence does not support and release the candidate.

        A refused CONFIRM must not leave the candidate active: an active candidate
        removes ``ANNOTATE_CANDIDATE`` from the UAV menu and the search would
        stall. REJECT is the non-terminal action that ends this verification round
        and resumes coverage without asserting a wrong match.
        """
        if "REJECT" not in obs.available_actions:
            return self._drop(fallback, reason)
        candidate = target.get("candidate_id")
        if not candidate and obs.active_candidate is not None:
            candidate = getattr(obs.active_candidate, "candidate_id", None)
        self.last_trace["arbitration_reason"] = reason
        self.last_trace["refused_claim"] = deepcopy(target)
        return LiteAction(obs.role, "REJECT", candidate_id=candidate, reason=reason)

    def _candidate_gap(self, obs):
        """Metres between the UGV and the centre of the candidate it wants to claim."""
        candidate = obs.active_candidate
        if candidate is None:
            return None
        centre = getattr(candidate, "region_center_xy", None)
        if not centre:
            return None
        position = obs.agent_position
        try:
            return math.hypot(float(position[0]) - float(centre[0]),
                              float(position[1]) - float(centre[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def _force_in_range_confirm(self, obs):
        """Turn a wasted wait at the candidate into the identity question.

        v10: the UGV passed within 30 m of the true target in 19 CONF-EASY
        episodes and answered STOP_OBSERVE (a 10 s wait) instead of CONFIRM. The
        confirm site is the only place identity is decidable, so the wait is a
        pure loss, and the 80-step contract rewards asking as soon as the
        geometry allows. The controller owns *when* to ask; the model still owns
        the answer, because the forced call routes through ``verify_confirm`` and
        a non-MATCH answer releases the candidate instead of confirming it.
        """
        if "CONFIRM" not in (obs.available_actions or ()):
            return None
        if obs.active_candidate is None:
            return None
        gap = self._candidate_gap(obs)
        if gap is None or gap > self.max_confirm_distance_m:
            return None
        self.last_trace["arbitration_reason"] = "in_range_confirm_forced"
        self.last_trace["in_range_gap_m"] = round(gap, 1)
        return {"action": "CONFIRM",
                "candidate_id": getattr(obs.active_candidate, "candidate_id", None)}

    def _proposal_reach(self, obs, proposals):
        """Straight-line UGV distance per proposal, so the model can prefer reachable ones.

        The model only sees its own frame; without this number it nominates cars the
        UGV cannot reach inside the step contract (v6: 603 m away, 80 steps, never
        arrived). ``ProposalScreen`` hands back plain dicts, while the projector wants
        a proposal object, so each item is wrapped with getattr-safe geometry.
        """
        ground = (getattr(self.coverage, "positions", {}) or {}).get("UGV")
        adapter = getattr(self.screen, "air", None)
        if ground is None or not proposals or adapter is None:
            return {}
        frame = getattr(obs, "uav_frame", None)
        reach = {}
        for proposal in proposals:
            item = proposal if isinstance(proposal, dict) else asdict(proposal)
            view = _ProposalView(item)
            if not view.bbox_xyxy:
                continue
            point = project_observation_proposal(adapter, obs, frame, view)
            if point is None:
                continue
            reach[_short_id(item.get("proposal_id"))] = round(
                math.dist(tuple(point)[:2], tuple(ground)[:2]), 1)
        return reach

    def _model_action(self, obs, decision, visual, fallback):
        if decision.get("status") == "budget_exhausted":
            # L1: the framework asked the model with an empty budget; that is an
            # accounting fact, not a model fault.
            return self._drop(fallback, "decision_budget_exhausted")
        target = decision.get("action") or {}
        action_name = target.get("action")
        if obs.role == "UGV" and action_name not in ("CONFIRM", "REJECT"):
            forced = self._force_in_range_confirm(obs)
            if forced is not None:
                target, action_name = forced, "CONFIRM"
        if obs.role == "UGV" and action_name in ("CONFIRM", "REJECT"):
            # 2026-10-31 v9: v4-v8 gated the claim on the distance to the candidate
            # centre, but the centre comes from an 80 m drone projection and can sit
            # 60-90 m from the road port the router actually stops at (v8: eight
            # arrivals, closest claim gap 64.7 m). The framework already owns a
            # serviceability-based arrival; the method defers to it and keeps the
            # identity question (the model's job) as the second gate. The 30 m success
            # rule is still enforced by the runner against the ground truth.
            #
            # 2026-10-31 v10: deferring *only* to the framework was the opposite
            # error. The drone point lands off-road, so the router's port can be
            # 619 m away while the UGV drives straight over the true target: in 19
            # of the 23 CONF-EASY episodes whose stop 6-view held the GT within
            # 30 m, the UGV issued zero confirms. Standing within the success
            # radius of the candidate is the same physical fact the runner
            # verifies with the stop 6-view, so it also opens the claim.
            route = self.last_trace.get("route") or {}
            phase = self.controller.roles[obs.role]["phase"]
            gap = self._candidate_gap(obs)
            near = gap is not None and gap <= self.max_confirm_distance_m
            arrived = (bool(route.get("arrived"))
                       or str(route.get("status")) in ("arrived", "AT_SERVICE_PORT")
                       or phase in ("arrival", "verify")
                       or near)
            if not arrived:
                self.last_trace["claim_gap_m"] = gap
                moving = getattr(fallback, "action", None) in ("FORWARD_1", "FORWARD_2", "REVERSE")
                if action_name == "CONFIRM" and not moving and "FORWARD_1" in obs.available_actions:
                    return self._drop(LiteAction("UGV", "FORWARD_1"), "claim_before_arrival")
                return self._drop(fallback, "claim_before_arrival")
        if obs.role == "UGV" and action_name == "CONFIRM":
            # 2026-10-31 P0 (conf238-agos_b): CONFIRM is legal at any step
            # (runner.compute_ugv_available_actions, user ruling 2026-09-04), so 86 of
            # 91 confirms were executed on a wrong vehicle while the model's own
            # verdict was UNKNOWN. The claim is now gated on a ground-resolution
            # identity question: refuse it and release the candidate instead.
            evidence = self.policy.verify_confirm(obs)
            if evidence.get("status") == "budget_exhausted":
                return self._drop(fallback, "verify_budget_exhausted")
            if evidence.get("target_verdict") != "MATCH":
                return self._drop_reject(obs, target, fallback,
                                         "confirm_without_match:" + str(evidence.get("target_verdict")))
        if obs.role == "UAV" and action_name == "SUSPECT_CANDIDATE":
            if "ANNOTATE_CANDIDATE" not in obs.available_actions:
                # M3': with an active candidate the menu drops ANNOTATE_CANDIDATE;
                # the suspect chain cannot publish, so it must not burn a verify call.
                return self._drop(fallback, "annotation_unavailable")
            verification = self.policy.verify_proposals(obs, visual["proposals"],
                                                         phase=self.controller.roles[obs.role]["phase"])
            if verification.get("status") == "budget_exhausted":
                return self._drop(fallback, "verify_budget_exhausted")
            verdict = verification.get("target_verdict")
            if verdict == "NON_MATCH":
                # M2': an explicit rejection is a different fact from "cannot judge".
                return self._drop(fallback, "proposal_non_match")
            # L10 (2026-10-31 P0): the crop verdict answered UNKNOWN on 90% of the
            # 2,248 nominations, and requiring MATCH here dropped 2,222 of them --
            # only 26 candidates ever reached the UGV in 238 episodes. A drone crop
            # at ~20 px cannot settle identity; publication therefore follows the
            # model's nomination and identity is decided at the ground confirm site.
            native = visual.get("native_result")
            selected = _short_id(verification.get("selected_proposal"))
            proposals = list(getattr(native, "proposals", ()) or ())
            proposal = next((p for p in proposals
                             if _short_id(getattr(p, "proposal_id", None)) == selected), None)
            if proposal is None:
                named = _short_id(target.get("proposal_id")) or _short_id(target.get("candidate_id"))
                proposal = next((p for p in proposals
                                 if _short_id(getattr(p, "proposal_id", None)) == named), None)
            if proposal is None:
                return self._drop(fallback, "proposal_not_in_menu")
            frame = getattr(obs, "uav_frame", None)
            point = project_observation_proposal(self.screen.air, obs, frame, proposal)
            if point is None:
                return self._drop(fallback, "proposal_unprojectable")
            ground = (getattr(self.coverage, "positions", {}) or {}).get("UGV")
            if ground is not None:
                reach = math.dist(tuple(point)[:2], tuple(ground)[:2])
                if reach > PUBLISH_MAX_UGV_DISTANCE_M:
                    self.last_trace["publish_reach_m"] = round(reach, 1)
                    return self._drop(fallback, "proposal_out_of_reach")
            return LiteAction("UAV", "ANNOTATE_CANDIDATE", geometry_type="point",
                              position_xy=tuple(point), localization_source="gsam_point",
                              source_frame_id=frame.frame_id,
                              source_image_sha256=hashlib.sha256(frame.image_bytes).hexdigest())
        if action_name not in obs.available_actions:
            return self._drop(fallback, "invalid_model_action")
        if action_name == "MOVE_TO_UAV_WAYPOINT":
            legal = {item.get("waypoint_id") if isinstance(item, dict) else item.waypoint_id
                     for item in obs.available_waypoints}
            waypoint = target.get("waypoint_id")
            if waypoint not in legal:
                # H2': the SYSTEM advertises fallback_waypoint_id; a legal
                # fallback is the model's own move, not a drop.
                waypoint = target.get("fallback_waypoint_id")
            if waypoint not in legal:
                return self._drop(fallback, "invalid_model_waypoint")
            if waypoint != target.get("waypoint_id"):
                target = dict(target, waypoint_id=waypoint)
        action_fields = {field.name for field in fields(LiteAction)} - {"role", "action"}
        action = LiteAction(obs.role, action_name,
                            **{key: value for key, value in target.items() if key in action_fields})
        # Semantic verdict/evidence are recorded as structured method evidence;
        # benchmark still validates the original legal CONFIRM/REJECT action.
        return action

    def _message(self, obs, decision, action, boundary, reply_applied=False):
        payload = None
        if reply_applied and decision.get("request_reply") in ("accept", "defer"):
            payload = {"type": "assistance_reply", "request_id": decision.get("request_id"),
                       "request_reply": decision["request_reply"]}
        elif action.action == "ANNOTATE_CANDIDATE" or (
                decision.get("request_assistance")
                and (obs.active_candidate is not None or action.candidate_id)):
            # A handoff must bind a public candidate/evidence; an unbound model
            # request for "assistance" is dropped instead of fabricating a task.
            # L8: bind the region to the public candidate centre first -- the
            # annotating UAV's own position is not where the UGV should go.
            if action.action == "ANNOTATE_CANDIDATE" and getattr(action, "position_xy", None):
                region = list(action.position_xy)
            elif obs.active_candidate is not None:
                region = list(obs.active_candidate.region_center_xy)
            else:
                region = list(getattr(action, "position_xy", None) or obs.agent_position[:2])
            payload = {"type": "assistance_request",
                       "request_id": f"{obs.episode_id}:{obs.role}:{obs.step}",
                       "region_xy": region,
                       "candidate_id": action.candidate_id,
                       "source_image_sha256": action.source_image_sha256}
        elif action.action in ("REJECT", "CONFIRM") and boundary.get("request"):
            payload = {"type": "verification_complete",
                       "request_id": boundary["request"]["request_id"]}
        if payload:
            action.message = json.dumps({"schema": "COOP_MESSAGE_V1", **payload}, separators=(",", ":"))
        return action

    @staticmethod
    def _private_target(context):
        plan = context.get("episode_plan") or context
        transform = (plan.get("target_oracle") or {}).get("transform") or {}
        try:
            x, y = float(transform["x"]), float(transform["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return {"target_xy": (x, y)}

    def _expert_public(self, obs, boundary, visual, fallback):
        public = {"role": obs.role, "task_phase": boundary["phase"],
                  "call_reason": boundary.get("call_reason"),
                  "availability_ready": boundary.get("action_ready", True),
                  "algorithm_action": asdict(fallback) if fallback is not None else {},
                  "request": deepcopy(boundary.get("request")),
                  "active_candidate": asdict(obs.active_candidate) if obs.active_candidate else {}}
        if obs.role == "UAV":
            frame = getattr(obs, "uav_frame", None)
            if frame is not None and frame.valid:
                public["uav_frame"] = {
                    "x": frame.x, "y": frame.y, "z": frame.z,
                    "yaw_deg": frame.yaw_deg, "fov_deg": frame.fov_deg,
                    "width": frame.width, "height": frame.height,
                    "frame_id": frame.frame_id,
                    "image_sha256": hashlib.sha256(frame.image_bytes).hexdigest()}
                projected = []
                native = visual.get("native_result")
                native_items = list(getattr(native, "proposals", ()) or ())
                for proposal in native_items:
                    point = project_observation_proposal(self.screen.air, obs, frame, proposal)
                    if point is None:
                        continue
                    projected.append({"proposal_id": getattr(proposal, "proposal_id", None),
                                      # Detector-frame box, the same space the
                                      # deployed prompt exposes via public_proposal;
                                      # shown here with the expert-only world_xy
                                      # projection so the expert can check the box
                                      # really covers the target pixel.
                                      "bbox_xyxy": list(getattr(proposal, "bbox_xyxy", ()) or ()),
                                      "world_xy": [float(point[0]), float(point[1])]})
                public["proposals"] = projected
        else:
            public["ugv"] = {
                "position": list(obs.agent_position[:2]),
                "yaw_deg": float(getattr(obs, "r18_ugv_yaw_deg", 0.0)),
                "camera_height_m": float(getattr(obs, "r18_camera_height_m", 2.2)),
                "fov_deg": float(self.cfg.get("ugv_view_fov_deg", 90.0)),
                "views": [{"yaw_offset_deg": view.yaw_offset_deg, "width": view.width,
                           "height": view.height}
                          for view in obs.ugv_views if view.valid]}
        return public

    def _expert_step(self, obs, boundary, visual, fallback, route, summary):
        """GT expert owns the model-side decision site; zero real model calls."""
        public = self._expert_public(obs, boundary, visual, fallback)
        decision = self.expert.decide(public, self._expert_gt)
        target = decision.get("target_action") or {}
        name = target.get("action")
        action = None
        if name == "ANNOTATE_CANDIDATE" and obs.role == "UAV":
            frame = getattr(obs, "uav_frame", None)
            current_sha = hashlib.sha256(frame.image_bytes).hexdigest() if frame is not None else None
            if (frame is not None and frame.valid
                    and str(target.get("source_frame_id")) == str(frame.frame_id)
                    and target.get("source_image_sha256") == current_sha
                    and "ANNOTATE_CANDIDATE" in obs.available_actions):
                action = LiteAction("UAV", "ANNOTATE_CANDIDATE", geometry_type="point",
                                    position_xy=tuple(target["position_xy"]),
                                    localization_source="expert_public_proposal",
                                    source_frame_id=frame.frame_id,
                                    source_image_sha256=current_sha)
        elif name and name in obs.available_actions:
            fields_allowed = {field.name for field in fields(LiteAction)} - {"role", "action"}
            action = LiteAction(obs.role, name,
                                **{key: value for key, value in target.items()
                                   if key in fields_allowed})
        control = {key: deepcopy(value) for key, value in decision.items()
                   if key in ("request_reply", "request_id", "resume_search")}
        # The control decision is applied even when the expert chose no action, so
        # arrival advances to verify and a deferral consumes its receipt exactly once.
        applied = self.controller.apply_decision(
            obs.role, {**control, "action": name} if name in ("CONFIRM", "REJECT") else control)
        self.last_trace["expert_decision"] = deepcopy(decision)
        self.last_trace["decision_owner"] = "expert"
        self.last_trace["control_applied"] = bool(applied)
        if action is None:
            self.last_trace["deployment_owner"] = "coverage_algorithm"
            phase = self.controller.roles[obs.role]["phase"]
            if applied and phase == "approach":
                route, approach = self._approach(obs)
                action = approach or self._rest(obs)
                if approach is not None:
                    self.last_trace["deployment_owner"] = "approach_router"
            elif applied and phase == "coverage":
                action = self.coverage.next_action(obs)
            else:
                action = fallback
            # H2' (audit): the delegated target must carry the full algorithm
            # action (waypoint_id included) or the student learns bare moves
            # that deployment drops as invalid_model_waypoint.
            record_target = {"owner_choice": "algorithm",
                             "action": asdict(action), **control}
        else:
            self.last_trace["deployment_owner"] = "expert"
            self._message(obs, {"action": target}, action, boundary)
            record_target = {"owner_choice": "mllm", "action": deepcopy(target), **control}
        # H1' (audit): the deployment path feeds coverage into the prompt via
        # public_state_summary; the expert site must record the same bytes.
        self._expert_capture(obs, boundary, visual, route, summary, public, record_target)
        self._step_calls = self.policy.drain_events()
        self.last_trace["method_proposal"] = asdict(action)
        return action

    def _expert_capture(self, obs, boundary, visual, route, summary, public, record_target):
        """Record this step's expert questions: decision, verify and ground confirm.

        Shared by ``drive`` and ``label`` modes so both produce byte-identical
        rows for the same question; only the actor differs.
        """
        self.policy.record_expert(obs, {**boundary, "coverage": summary, "route": route,
                                   "proposal_reach": self._proposal_reach(obs, visual.get("proposals") or [])},
                                  visual.get("proposals") or [], boundary.get("call_reason"),
                                  record_target)
        if obs.role == "UAV" and visual.get("proposals"):
            # Deployment asks a second, narrower question whenever the student
            # commits to a proposal; the expert must label that same site or the
            # adapter is deployed onto a duty it never saw. M5: only a non-empty
            # board makes that question real; empty boards are never asked.
            label = self.expert.verify_label(public, self._expert_gt)
            if label is not None:
                self.policy.record_expert_verify(obs, visual.get("proposals") or [], label,
                                                 boundary.get("call_reason"))
        # 2026-11-01: the ground confirmation site. Deployment asks
        # ``verify_confirm`` once the UGV stands at the candidate; without this
        # record no training row ever carries that question, and it is the one
        # site where the student fails (v10: 4/7 true targets matched).
        if obs.role == "UGV" and getattr(self.expert, "confirm_capture", False):
            candidate_id = getattr(obs.active_candidate, "candidate_id", None)
            gap = self._candidate_gap(obs)
            in_range = gap is not None and gap <= self.max_confirm_distance_m
            phase_now = self.controller.roles[obs.role]["phase"]
            if (candidate_id and candidate_id not in self._confirm_recorded
                    and (phase_now in ("arrival", "verify") or in_range)):
                verdict = self.expert.confirm_verdict(public, self._expert_gt)
                if verdict is not None:
                    # One row per candidate: the site is the arrival judgement,
                    # not every step spent waiting next to the vehicle.
                    self._confirm_recorded.add(candidate_id)
                    self.last_trace["expert_confirm"] = deepcopy(verdict)
                    self.policy.record_expert_confirm(obs, verdict, boundary.get("call_reason"))

    def _expert_label(self, obs, boundary, visual, fallback, route, summary):
        """Label-only mode: the student acts, the oracle annotates the same question.

        The point is the ground-confirmation site. While the expert drives, it only
        ever walks up to candidates near the GT, so every confirm row it produces is
        a ``MATCH`` (v11 pool: 25 MATCH, 0 NON_MATCH) and the row pool cannot teach
        rejection at all. Here the *model's* candidate brings the UGV to whatever
        vehicle it picked; the oracle's verdict at that spot is the label we could
        never collect otherwise (2026-11-01 §26.2).
        """
        if self._expert_gt is None:
            return
        public = self._expert_public(obs, boundary, visual, fallback)
        decision = self.expert.decide(public, self._expert_gt)
        target = decision.get("target_action") or {}
        # Deliberately no ``expert_decision`` on the trace and no control applied:
        # the student owns the action, the expert only supplies the label. Capture
        # marks a step "expert action also carries real calls" when both appear.
        self.last_trace["expert_label"] = deepcopy(decision)
        self._expert_capture(obs, boundary, visual, route, summary, public,
                             {"owner_choice": "mllm", "action": deepcopy(target)})

    def decide(self, observation):
        if self.coverage is None:
            raise ValueError("cooperative_search: episode not initialized")
        self._step_calls = self.policy.drain_events()
        obs = public_observation(observation)
        self._sample(obs)
        self._ingest_messages(obs)
        summary = self.coverage.summary(obs.role, float(getattr(obs, "coop_event_time", 0)))
        visual = self.screen.screen(obs)
        route, approach_action = self._approach(obs)
        route.update(evidence_id=visual.get("evidence_id"), fault_id=visual.get("fault_id") or route.get("fault_id"),
                     # L5: joint reallocation can grow the assigned set past the
                     # frozen initial count; clamp the progress denominator.
                     progress=len(summary["completed_units"]) /
                     max(1, summary["initial_count"], len(summary["assigned_units"])))
        boundary = self.controller.boundary(obs.role, obs, route)
        if not boundary["action_ready"]:
            raise ValueError("cooperative_search: decide before complete action boundary")
        if boundary["reallocate_reason"]:
            self.coverage.reallocate(boundary["reallocate_reason"], self.coverage.positions,
                                     float(getattr(obs, "coop_event_time", 0)))
        self.last_trace = {"method_id": self.method_id, "variant_id": self.cfg["arm"],
                           "revision": REVISION, "phase": boundary["phase"],
                           "call_reason": boundary["call_reason"], "role": obs.role,
                           "coverage": summary, "route": route, "vision": {
                               key: visual.get(key) for key in ("status", "error", "source", "evidence_id", "fault_id")},
                           "decision_owner": "framework", "deployment_owner": "coverage_algorithm"}
        fallback = approach_action or (self._rest(obs) if boundary["phase"] in ("receive", "arrival", "verify")
                                       else self.coverage.next_action(obs))
        if not boundary["call_reason"]:
            action = fallback
            if boundary["sender_policy"] == "hold" and boundary["phase"] == "coverage":
                action = self._rest(obs)
                self.last_trace["deployment_owner"] = "framework"
            elif approach_action:
                self.last_trace["deployment_owner"] = "approach_router"
            return action
        label_mode = self.expert is not None and getattr(self.expert, "label_only", False)
        if self.expert is not None and not label_mode:
            return self._expert_step(obs, boundary, visual, fallback, route, summary)
        state = {**boundary, "coverage": summary, "route": route,
                 "proposal_reach": self._proposal_reach(obs, visual.get("proposals") or [])}
        decision = self.policy.decide(obs, state, visual["proposals"], boundary["call_reason"],
                                      int(getattr(obs, "internal_vlm_calls_remaining", 0)))
        self.last_raw_response, self.last_parse_status = self.policy.last_raw_response, self.policy.last_parse_status
        self.last_trace.update(decision_owner="model", model_decision=deepcopy(decision))
        action = self._model_action(obs, decision, visual, fallback)
        owner_algorithm = (decision.get("owner_choice") == "algorithm" and boundary["phase"] == "coverage"
                           and (decision.get("action") or {}).get("action") not in ("CONFIRM", "REJECT", "SUSPECT_CANDIDATE"))
        if owner_algorithm:
            action = fallback
        self.last_trace["deployment_owner"] = "coverage_algorithm" if owner_algorithm else "model"
        # Terminal lifecycle follows the actual execution receipt, not the raw model string.
        control = {key: value for key, value in decision.items() if key != "action"}
        applied_reply = self.controller.apply_decision(obs.role, control)
        self.last_trace["control_applied"] = bool(applied_reply)
        self._message(obs, decision, action, boundary, reply_applied=applied_reply)
        if label_mode:
            self._expert_label(obs, boundary, visual, fallback, route, summary)
        self._step_calls = self.policy.drain_events()
        self.last_trace["method_proposal"] = asdict(action)
        return action

    def on_transition(self, observation, proposed_action=None, executed_action=None, execution_result=None):
        receipt = public_transition(execution_result or {})
        self.controller.transition(observation.role, receipt)
        self.coverage.observe(observation, executed_action, receipt)
        if executed_action is not None and executed_action.action in ("CONFIRM", "REJECT"):
            self.controller.apply_decision(observation.role, {"action": executed_action.action})
        self.last_trace["execution"] = receipt

    def build_trace(self, obs):
        trace = deepcopy(self.last_trace)
        trace["model_calls"] = deepcopy(self._step_calls)
        return trace
