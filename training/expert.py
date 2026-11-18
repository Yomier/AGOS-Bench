"""Ground-truth expert for collecting labels at deployed decision sites."""
from __future__ import annotations

import math

INF = float("inf")



def _xy(value):
    return (float(value[0]), float(value[1]))


class EventExpert:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.confirm_distance_m = float(config.get("confirm_distance_m", 20.0))
        self.annotate_proposal_max_m = float(config.get("annotate_proposal_max_m", 30.0))
        self.min_views = int(config.get("min_views", 1))
        self.view_margin = float(config.get("view_margin", 0.9))
        self.confirm_capture = bool(config.get("confirm_capture", False))
        self.mode = str(config.get("mode", "drive"))
        if self.mode not in ("drive", "label"):
            raise ValueError("cooperative_search.expert.mode must be drive or label")
        self.label_only = self.mode == "label"

    # ---- public geometry helpers -------------------------------------
    @staticmethod
    def _bearing_deg(dx, dy):
        return math.degrees(math.atan2(dy, dx)) % 360.0

    def _uav_projection(self, target, frame):
        """Return ``(inside, u, v)`` using the runtime's camera model.

        The runtime projects through ``uav_grid_geometry.world_to_pixel``, whose
        ground footprint is aspect-corrected. Assuming a square footprint here
        put the vertical ray off by the aspect ratio, so the expert's "is the
        target in this frame" gate disagreed with what the student sees.
        """
        width, height = frame.get("width"), frame.get("height")
        if not width or not height:
            return False, 0.0, 0.0
        from agos.perception.uav_grid_geometry import world_to_pixel

        px, py = world_to_pixel(
            float(frame["x"]), float(frame["y"]), float(frame["z"]),
            float(frame.get("yaw_deg", 0.0)), float(frame["fov_deg"]),
            int(width), int(height), float(target[0]), float(target[1]))
        if px is None:
            return False, 0.0, 0.0
        u, v = float(px) / float(width), float(py) / float(height)
        margin = (1.0 - self.view_margin) / 2.0
        inside = margin <= u <= 1.0 - margin and margin <= v <= 1.0 - margin
        return inside, u, v

    def _ugv_visible_views(self, target, ugv):
        x, y = _xy(ugv["position"])
        yaw = float(ugv.get("yaw_deg", 0.0))
        height = float(ugv.get("camera_height_m", 2.2))
        fov = float(ugv.get("fov_deg", 90.0))
        views, seen = ugv.get("views") or [], 0
        dx, dy = target[0] - x, target[1] - y
        distance = math.hypot(dx, dy)
        if distance <= 1e-6 or distance > 60.0:
            return 0, distance
        bearing = self._bearing_deg(dx, dy)
        for view in views:
            view_yaw = (yaw + float(view.get("yaw_offset_deg", 0.0))) % 360.0
            delta = abs((bearing - view_yaw + 180.0) % 360.0 - 180.0)
            if delta > fov / 2.0 * self.view_margin:
                continue
            width = float(view.get("width") or 480)
            height_px = float(view.get("height") or 270)
            vfov = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov) / 2.0) * height_px / width))
            elevation = math.degrees(math.atan2(height, distance))
            if abs(elevation) > vfov / 2.0:
                continue
            seen += 1
        return seen, distance

    # ---- public decision surface -------------------------------------
    def decide(self, public: dict, private: dict) -> dict:
        """Return {target_action, expert_authority, observable_support, eligibility}."""
        target = _xy(private["target_xy"])
        role = str(public.get("role", "")).upper()
        phase = str(public.get("task_phase", "coverage"))
        if public.get("availability_ready") is False:
            return self._hold(role, phase, "wait_public_ready")
        if phase in ("receive",):
            return self._receipt(public, target)
        if role == "UAV":
            return self._uav(public, target, phase)
        return self._ugv(public, target, phase)

    def _hold(self, role, phase, why):
        return {"target_action": {}, "expert_authority": "gt_expert",
                "observable_support": {"held": why}, "eligibility": "hold",
                "fallback": "algorithm"}

    def verify_label(self, public, private):
        """GT label for the deployment's crop-verification call, or ``None``.

        Deployment asks a second, narrower question: which shown proposal is the
        reference vehicle. The expert answers from the same public evidence the
        student sees. An in-frame target with no proposal covering its pixel is a
        legitimate ``NONE``; a target outside the frame yields no label, because
        nothing about the shown proposals can be judged from it.
        """
        frame = public.get("uav_frame")
        proposals = public.get("proposals") or []
        if not frame or not proposals:
            return None
        target = _xy(private["target_xy"])
        inside, u, v = self._uav_projection(target, frame)
        if not inside:
            return None
        width, height = float(frame["width"]), float(frame["height"])
        on_target = self._proposal_on_target(proposals, u * width, v * height)
        if on_target is not None and on_target.get("world_xy"):
            distance = math.hypot(on_target["world_xy"][0] - target[0],
                                  on_target["world_xy"][1] - target[1])
            if distance <= self.annotate_proposal_max_m:
                return {"selected_proposal": on_target.get("proposal_id"),
                        "target_verdict": "MATCH"}
        return {"selected_proposal": "NONE", "target_verdict": "UNKNOWN"}

    def confirm_verdict(self, public, private):
        """Judge a distant candidate or a nearby target visible in the ground views."""
        candidate = public.get("active_candidate") or {}
        center = candidate.get("region_center_xy")
        if not center:
            return None
        target = _xy(private["target_xy"])
        gap = math.hypot(float(center[0]) - target[0], float(center[1]) - target[1])
        ugv = public.get("ugv") or {}
        if not ugv.get("position"):
            return None
        seen, distance = self._ugv_visible_views(target, ugv)
        support = {"candidate_id": candidate.get("candidate_id"),
                   "candidate_center_distance_m": round(gap, 3),
                   "target_distance_m": round(distance, 3),
                   "visible_views": seen}
        if gap > self.annotate_proposal_max_m:
            return {"target_verdict": "NON_MATCH", "observable_support": support}
        if distance <= self.confirm_distance_m and seen >= self.min_views:
            return {"target_verdict": "MATCH", "observable_support": support}
        return None

    def _receipt(self, public, target):
        """Judge the candidate centre published on the active candidate or request."""
        request = public.get("request") or {}
        candidate = public.get("active_candidate") or {}
        support = {"candidate_id": candidate.get("candidate_id"),
                   "request_id": request.get("request_id"),
                   "accept_radius_m": self.annotate_proposal_max_m}
        center = candidate.get("region_center_xy")
        if not center:
            center = request.get("region_xy")
            if center:
                support["region_source"] = "request_region_xy"
        if not center:
            support["held"] = "candidate_without_public_region"
            return {"target_action": {}, "request_reply": "defer",
                    "request_id": request.get("request_id"),
                    "expert_authority": "gt_expert", "observable_support": support,
                    "eligibility": "defer_no_public_region"}
        distance = math.hypot(float(center[0]) - target[0], float(center[1]) - target[1])
        support["candidate_distance_m"] = round(distance, 3)
        if distance > self.annotate_proposal_max_m:
            support["held"] = "public_candidate_far_from_target"
            return {"target_action": {}, "request_reply": "defer",
                    "request_id": request.get("request_id"),
                    "expert_authority": "gt_expert", "observable_support": support,
                    "eligibility": "defer_unrelated_candidate"}
        return {"target_action": {}, "request_reply": "accept",
                "request_id": request.get("request_id"),
                "expert_authority": "gt_expert", "observable_support": support,
                "eligibility": "accept_ground_confirmation"}

    def _proposal_on_target(self, proposals, u, v):
        """Return the smallest shown box containing the projected ground-truth pixel."""
        best, best_area = None, INF
        for proposal in proposals:
            box = proposal.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            if not (float(box[0]) <= u <= float(box[2]) and float(box[1]) <= v <= float(box[3])):
                continue
            area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
            if area < best_area:
                best, best_area = proposal, area
        return best

    def _uav(self, public, target, phase):
        frame = public.get("uav_frame")
        proposals = public.get("proposals") or []
        if not frame or not proposals:
            return self._hold("UAV", phase, "no_public_evidence")
        inside, u, v = self._uav_projection(target, frame)
        if not inside:
            return self._hold("UAV", phase, "target_outside_current_frame")
        width, height = float(frame["width"]), float(frame["height"])
        # World metres and frame pixels are different spaces; the label is only
        # imitable when the chosen box contains the target pixel itself.
        on_target = self._proposal_on_target(proposals, u * width, v * height)
        if on_target is None or not on_target.get("world_xy"):
            return self._hold("UAV", phase, "no_public_proposal_on_target")
        distance = math.hypot(on_target["world_xy"][0] - target[0],
                              on_target["world_xy"][1] - target[1])
        if distance > self.annotate_proposal_max_m:
            return self._hold("UAV", phase, "no_public_proposal_on_target")
        nearest, best = on_target, distance
        action = {"role": "UAV", "action": "ANNOTATE_CANDIDATE", "geometry_type": "point",
                  "position_xy": [float(nearest["world_xy"][0]), float(nearest["world_xy"][1])],
                  "localization_source": "expert_public_proposal",
                  "source_frame_id": frame.get("frame_id"),
                  "source_image_sha256": frame.get("image_sha256", "")}
        return {"target_action": action, "expert_authority": "gt_expert",
                "observable_support": {"proposal_id": nearest.get("proposal_id"),
                                       "proposal_distance_m": round(best, 3),
                                       "frame_u": round(u, 4), "frame_v": round(v, 4)},
                "eligibility": "annotate_public_proposal"}

    def _ugv(self, public, target, phase):
        ugv = public.get("ugv") or {}
        seen, distance = self._ugv_visible_views(target, ugv)
        support = {"visible_views": seen, "distance_m": round(distance, 3),
                   "phase": phase}
        if distance > self.confirm_distance_m or seen < self.min_views:
            if phase in ("arrival", "verify"):
                support["held"] = "insufficient_gt_support"
                if phase == "verify":
                    # Leaving an unknown candidate is not a REJECT claim: resume the
                    # public coverage task instead of asserting NON_MATCH.
                    return {"target_action": {}, "resume_search": True,
                            "expert_authority": "gt_expert", "observable_support": support,
                            "eligibility": "resume_unverified_candidate"}
                return {"target_action": {}, "expert_authority": "gt_expert",
                        "observable_support": support, "eligibility": "arrival_observe"}
            return self._hold("UGV", phase, "insufficient_gt_support")
        active = public.get("active_candidate") or {}
        action = {"role": "UGV", "action": "CONFIRM",
                  "candidate_id": active.get("candidate_id", "") or "",
                  "target_verdict": "MATCH"}
        return {"target_action": action, "expert_authority": "gt_expert",
                "observable_support": support, "eligibility": "confirm_visible_target"}

    # ---- coverage fallback (no GT influence) -------------------------
    @staticmethod
    def coverage_action(public):
        return public.get("algorithm_action") or {}
