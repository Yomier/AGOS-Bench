"""Public frame proposals and viewpoint-aware events; never identity gold."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math

from PIL import Image

from agos.methods.common.grounded_localizer import GroundedLocalizer
from agos.methods.common.ugv_sixview import UgvGroundedEvidenceClient, build_evidence_bundle


class ProposalScreen:
    def __init__(self, config: dict):
        self.config = config
        self.air = GroundedLocalizer(
            config.get("gsam_endpoint", "http://127.0.0.1:8201/v1/gsam"),
            timeout_s=config.get("tool_timeout_s", 120),
            box_threshold=config.get("box_threshold", 0.2),
            text_threshold=config.get("text_threshold", 0.18), max_proposals=3)
        self.ground = UgvGroundedEvidenceClient(
            config.get("ugv_endpoint", "http://127.0.0.1:8201/v1/ugv_views"),
            timeout_s=config.get("tool_timeout_s", 120), display_limit=3,
            evidence_id_mode="local_card_v1")
        self.seen = set()
        self.threshold = float(config.get("wake_identity_min", 0.72))
        self.viewpoint_m = float(config.get("evidence_viewpoint_m", 20))
        if not math.isfinite(self.viewpoint_m) or self.viewpoint_m <= 0:
            raise ValueError("cooperative_search: evidence viewpoint spacing must be positive")

    def _event(self, obs, proposals):
        pose_bin = tuple(math.floor(float(value) / self.viewpoint_m) for value in obs.agent_position[:2])
        yaw_bin = math.floor(float(getattr(obs, "r18_ugv_yaw_deg", 0)) / 30)
        new = []
        for proposal in proposals:
            score = float(proposal.get("identity_score", 0))
            if score < self.threshold:
                continue
            # Frame IDs and per-bundle proposal IDs must not make identical
            # evidence wake again. A meaningfully new viewpoint/box/score may.
            box = tuple(round(float(value) / 16) for value in proposal.get("bbox_xyxy", ()))
            key = (obs.role, pose_bin, yaw_bin, proposal.get("view_id"), box, int(score / 0.05))
            if key not in self.seen:
                self.seen.add(key)
                new.append(key)
        if not new:
            return None
        return hashlib.sha256(json.dumps(sorted(new, key=str)).encode("utf-8")).hexdigest()

    @staticmethod
    def _ground_crops(obs, proposals):
        views = {view.view_id: view for view in obs.ugv_views}
        output = []
        for proposal in proposals:
            if not proposal.get("shown", True):
                continue
            clean = {key: proposal[key] for key in
                     ("proposal_id", "view_id", "bbox_xyxy", "identity_score", "detector_score")
                     if key in proposal}
            view = views.get(clean.get("view_id"))
            box = clean.get("bbox_xyxy")
            if view is None or not box or len(box) != 4:
                raise ValueError("cooperative_search: invalid_source ground proposal/view")
            image = Image.open(io.BytesIO(view.image_bytes)).convert("RGB")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError("cooperative_search: invalid_source ground crop geometry")
            crop = image.crop((int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            clean["crop_b64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            output.append(clean)
        return output[:3]

    def screen(self, obs) -> dict:
        if obs.role == "UAV":
            result = self.air.ground_frame(getattr(obs, "uav_frame", None),
                                           obs.target_description, obs.target_reference_image,
                                           segment=False)
            status, error = result.status, result.error
            proposals = [proposal.to_dict() for proposal in result.proposals]
            source = {"frame_id": result.frame_id, "image_sha256": result.source_image_sha256}
            native = result
        else:
            bundle = build_evidence_bundle(obs)
            evidence = self.ground.request_board(bundle, obs.target_reference_image)
            status, error = evidence.get("status", "error"), evidence.get("error", "")
            proposals = self._ground_crops(obs, evidence.get("proposals") or []) if status in ("ok", "empty") else []
            source = {"bundle_id": bundle.bundle_id, "image_sha256": list(bundle.view_hashes())}
            native = evidence
        if status in ("invalid_source", "invalid_frame"):
            raise ValueError(f"cooperative_search: {status}: {error}")
        evidence_id = self._event(obs, proposals) if status in ("ok", "empty") else None
        fault_id = None
        if status not in ("ok", "empty"):
            # Repeated unchanged faults wake once in the controller, while each
            # tool attempt remains a recorded fault with its own source.
            fault_id = hashlib.sha256(f"{obs.role}:{status}:{error}".encode()).hexdigest()
        return {"status": status, "error": error, "proposals": proposals,
                "evidence_id": evidence_id, "fault_id": fault_id,
                "source": source, "native_result": native}
