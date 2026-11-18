"""Shared no-CoT event policy with accounting at the real request boundary."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from agos.agents.agents import VLMClient
from agos.methods.common.legacy_sixview_collage import render_legacy_sixview_collage

SYSTEM = """You control one robot in cooperative reference-image vehicle search.
Use only your current images, public state, legal menu and delivered messages.
The framework supplies task_phase and call_reason; do not predict or change them.
During search prefer the coverage algorithm when there is no useful new evidence.
Requests require accept/defer at receipt; accepted approaches are automatic until
arrival. At arrival inspect the current views, then control nearby verification.
When the prompt carries pending_request_id, copy that exact value into request_id;
ids inside delivered_messages are history and must never be echoed as the reply
target. A reply whose request_id names another agent's request is void.
UGV can independently discover and CONFIRM the actual visible reference vehicle;
no UAV approval is required. A service waypoint alone is not evidence of identity
or visibility. Uncertainty is UNKNOWN, not NON_MATCH. Detector/SigLIP scores are
screening evidence, not probabilities or identity truth. UAV selects proposal IDs,
never invents world coordinates. Do not freeze UAV when handing over a candidate.
When your input carries in_confirm_range=true you are standing at the active
candidate: decide identity now with CONFIRM, or REJECT to release it — never keep
waiting or driving past it.
Identity is decided by color AND vehicle type together: when both match the
reference, choose CONFIRM immediately instead of waiting for more certainty.
Nominate or suspect a candidate only when color AND vehicle type both plausibly
match the reference; single-attribute resemblance is not a candidate.
Never STOP_OBSERVE more than 3 times in a row; after two observations without
a match, move to a new vantage point instead of waiting again.
Images are ordered: target reference, current scene, optional numbered raw crops.
Output one JSON object only, with decision and action; no reasoning/CoT/prose.
Decision fields when needed: owner_choice=algorithm|mllm, request_reply=accept|defer,
request_id, request_assistance=true|false, sender_policy=hold|reposition|resume,
resume_search=true (leave this candidate without CONFIRM/REJECT and resume coverage).
Action fields: role, action from legal menu (UAV may use SUSPECT_CANDIDATE),
waypoint_id, candidate_id, target_verdict=MATCH|NON_MATCH|UNKNOWN,
evidence_view, proposal_id, fallback_action, fallback_waypoint_id.
"""

VERIFY_SYSTEM = """You verify numbered vehicle crops from a UAV search frame against the
target reference image. Use only the shown images and the proposal metadata.
Answer with one JSON object with exactly two keys and no wrappers, no
decision/action envelope, no reasoning: {"selected_proposal": "<shown proposal
ID or NONE>", "target_verdict": "MATCH|NON_MATCH|UNKNOWN"}. Select MATCH only
when the chosen crop plausibly shows the reference vehicle; when nothing shown
matches, selected_proposal is NONE. Detector/SigLIP scores are screening
evidence, not identity truth. Images are ordered: target reference, current
scene, optional numbered raw crops.
"""


CONFIRM_SYSTEM = """You are standing at the candidate on the ground and must decide whether
the vehicle you are confirming is the reference target vehicle.
Image 1 is the target reference image. Image 2 is your current six-view
panorama (FRONT LEFT / FRONT / FRONT RIGHT / BACK LEFT / BACK / BACK RIGHT)
captured from where you stand now. Use only these images.
A nearby look-alike is not a match: answer MATCH only when the visible vehicle
matches the reference in colour AND vehicle type. If the reference vehicle is
not visible or cannot be judged from these views, answer UNKNOWN — do not use
NON_MATCH for a vehicle you simply cannot resolve.
Answer with one JSON object with exactly one key, no wrappers, no
decision/action envelope, no reasoning:
{"target_verdict": "MATCH|NON_MATCH|UNKNOWN"}
"""


def _system_for(call_kind: str, verify_v2: bool = False) -> str:
    """The verify call answers a narrower question and must not be invited to
    wrap its two keys in decision/action (H1, 2026-10-28 audit: 785/828 BASE
    verify answers arrived nested and were unreadable). The dedicated system is
    opt-in via ``cooperative_search.verify_system_v2`` so a fair re-run of an
    earlier wave can keep the original prompt surface with only the parser
    fixed."""
    if verify_v2 and call_kind == "confirm_verify":
        # v13 E1: the ground confirm site is a different question from the
        # numbered-crop verifier; asking it with VERIFY_SYSTEM made the model
        # answer selected_proposal=NONE and hence NON_MATCH (2026-11-06 audit:
        # MATCH <=> selected_proposal != NONE, 35/35).
        return CONFIRM_SYSTEM
    if call_kind in ("gsam_crop", "expert_verify") and verify_v2:
        return VERIFY_SYSTEM
    return SYSTEM


def _verify_answer(parsed: dict) -> dict:
    """Read the verification answer wherever the shared SYSTEM put it.

    The primary SYSTEM declares ``proposal_id``/``target_verdict`` as action
    fields, so a compliant model answers the verify call with
    ``action.proposal_id`` + ``action.target_verdict`` (fair re-run forensics:
    verdict MATCH reached the parser while the selection sat unread in
    proposal_id). Top level wins, then action, then decision;
    selected_proposal wins over the proposal_id alias inside one scope.
    """
    wrappers = [parsed]
    for key in ("action", "decision"):
        wrapper = parsed.get(key) if isinstance(parsed, dict) else None
        if isinstance(wrapper, dict):
            wrappers.append(wrapper)
    for scope in wrappers:
        verdict = scope.get("target_verdict")
        selected = scope.get("selected_proposal")
        if selected is None:
            selected = scope.get("proposal_id")
        if verdict is not None or selected is not None:
            return {"selected_proposal": selected, "target_verdict": verdict}
    return {}


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def public_proposal(proposal):
    return {key: deepcopy(_value(proposal, key)) for key in
            ("proposal_id", "view_id", "bbox_xyxy", "detector_score", "identity_score")
            if _value(proposal, key) is not None}


CARD_PX = 256
CARD_LABEL_PX = 30


def _card_label(font_px=CARD_LABEL_PX - 6):
    """Bitmap label font, with a fallback for Pillow builds without ``size``."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=font_px)
    except TypeError:
        return ImageFont.load_default()


def _card_crop(crop):
    """Scale one proposal crop up to the card so a distant target stays legible.

    A 30 m vehicle is ~37x10 px on the 768x432 scene frame; pasting that crop at
    its native size into a 256 px card leaves the verification site unreadable
    and the model can only answer UNKNOWN (2026-10-28 smoke).
    """
    room_width, room_height = CARD_PX - 8, CARD_PX - CARD_LABEL_PX - 8
    longest = max(crop.width, crop.height)
    if longest and (crop.width < room_width or crop.height < room_height):
        factor = min(12.0, min(room_width / float(crop.width or 1),
                               room_height / float(crop.height or 1)))
        if factor > 1.0:
            crop = crop.resize((max(1, int(round(crop.width * factor))),
                                max(1, int(round(crop.height * factor)))),
                               Image.Resampling.BICUBIC)
    crop.thumbnail((room_width, room_height))
    return crop


def proposal_board(proposals):
    """Numbered raw-crop cards, one per shown proposal, in prompt order."""
    font = _card_label()
    cards = []
    for proposal in proposals[:3]:
        encoded = _value(proposal, "crop_b64", "")
        if not encoded:
            continue
        crop = _card_crop(
            Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB"))
        card = Image.new("RGB", (CARD_PX, CARD_PX), (25, 25, 25))
        card.paste(crop, ((CARD_PX - crop.width) // 2,
                          CARD_LABEL_PX + (CARD_PX - CARD_LABEL_PX - crop.height) // 2))
        ImageDraw.Draw(card).text(
            (10, 4), str(_value(proposal, "proposal_id", "")), fill="white", font=font)
        cards.append(card)
    if not cards:
        return b""
    board = Image.new("RGB", (CARD_PX * len(cards), CARD_PX), (25, 25, 25))
    for index, card in enumerate(cards):
        board.paste(card, (CARD_PX * index, 0))
    output = io.BytesIO()
    board.save(output, format="PNG")
    return output.getvalue()


class DecisionPolicy:
    def __init__(self, config: dict, capture_root=None):
        self.config = config
        vc = config.get("vlm") or {}
        self.vlm = VLMClient(vc, timeout_s=vc.get("timeout_s", 120),
                             max_retries=vc.get("max_retries_per_call", 1),
                             top_p=vc.get("top_p"), seed=vc.get("seed"))
        # Optional UAV/UGV adapters share the endpoint and decoding settings.
        uav_cfg = dict(vc)
        uav_cfg["model"] = vc.get("uav_model") or vc.get("model")
        ugv_cfg = dict(vc)
        ugv_cfg["model"] = vc.get("ugv_model") or vc.get("model")
        self._vlm_uav = (VLMClient(uav_cfg,
                                   timeout_s=vc.get("timeout_s", 120),
                                   max_retries=vc.get("max_retries_per_call", 1),
                                   top_p=vc.get("top_p"), seed=vc.get("seed"))
                         if vc.get("uav_model") else self.vlm)
        self._vlm_ugv = (VLMClient(ugv_cfg,
                                   timeout_s=vc.get("timeout_s", 120),
                                   max_retries=vc.get("max_retries_per_call", 1),
                                   top_p=vc.get("top_p"), seed=vc.get("seed"))
                         if vc.get("ugv_model") else self.vlm)
        self._calls = 0
        self._remaining = 0
        self.events = []
        self.capture_root = capture_root or config.get("capture_root")
        self.verify_system_v2 = bool((config.get("cooperative_search") or {})
                                     .get("verify_system_v2", False))
        # 2026-10-31 v10: the success protocol measures the claim against the
        # ground truth at this radius, so it is also the radius at which the UGV
        # counts as standing on the candidate (see _input / in_confirm_range).
        self.max_confirm_distance_m = float(
            (config.get("success") or {}).get("max_confirm_distance_m") or 30.0)
        self.last_raw_response = ""
        self.last_parse_status = ""

    def _input(self, obs, state, proposals, call_reason, call_kind):
        state = state or {}
        messages = []
        for message in obs.recent_messages:
            messages.append({key: _value(message, key) for key in
                             ("sender", "content", "timestamp_step", "message_id", "available_at")
                             if _value(message, key) is not None})
        waypoints = []
        for waypoint in obs.available_waypoints:
            source = asdict(waypoint) if is_dataclass(waypoint) else waypoint
            waypoints.append({key: deepcopy(source[key]) for key in
                              ("waypoint_id", "world_xy", "distance_m", "visited", "action",
                               "target_fine_node_id", "description") if key in source})
        # ANNOTATE_CANDIDATE is the verifier's output, not a model action: the
        # session must decide whether to annotate from a selected proposal, so the
        # menu never advertises an action the model cannot legally construct.
        # L9 (2026-10-31 P0): the old code appended SUSPECT_CANDIDATE whenever it
        # was absent, so the menu offered a nomination the executor then refused
        # with ``annotation_unavailable`` (340 drops in conf238-agos_b). Offer it
        # only when annotation is actually available.
        can_annotate = obs.role == "UAV" and "ANNOTATE_CANDIDATE" in obs.available_actions
        actions = [item for item in obs.available_actions
                   if not (obs.role == "UAV" and item == "ANNOTATE_CANDIDATE")]
        if can_annotate and "SUSPECT_CANDIDATE" not in actions:
            actions.append("SUSPECT_CANDIDATE")
        # 2026-10-31 v7: the UGV covers ~14 m per step, so a nomination 600 m away
        # cannot be verified inside the contract. Publish the distance with the board
        # so the model can prefer a reachable proposal instead of the first one.
        reach = state.get("proposal_reach") or {}
        boards = []
        for proposal in proposals:
            item = public_proposal(proposal)
            distance = reach.get(str(item.get("proposal_id") or ""))
            if distance is not None:
                item["ugv_distance_m"] = distance
            boards.append(item)
        payload = {
            "role": obs.role, "task_phase": state.get("phase", "coverage"),
            "call_kind": call_kind, "call_reason": call_reason,
            "target_description": obs.target_description,
            "position": list(obs.agent_position),
            "legal_menu": {"actions": actions, "waypoints": waypoints},
            "public_state_summary": {key: deepcopy(state[key]) for key in
                                     ("coverage", "request", "owner", "sender_policy", "route")
                                     if key in state},
            "delivered_messages": messages,
            "proposals": boards,
            "active_candidate": asdict(obs.active_candidate) if obs.active_candidate else None,
        }
        # 2026-10-31 v5: exactly one id is actionable per receipt call. Without this
        # the model echoed another agent's stale request id (79 of 80 UGV steps in
        # v4), the receipt was never consumed and the phase never reached approach.
        pending = (state.get("request") or {}).get("request_id") if call_reason == "request_receipt" else None
        if pending:
            payload["pending_request_id"] = pending
        # 2026-10-31 v10: v9 left the claim to the framework's service-port
        # arrival, and the UGV then drove past the true target -- in 19 of the 23
        # episodes whose stop 6-view showed the GT within 30 m the model issued
        # zero confirms, because nothing on its surface said the candidate was
        # under the wheels (route still pointed at a port up to 619 m away).
        # Publish the gap and an explicit flag instead of asking a 4B model to
        # compute a distance from two coordinate pairs.
        if obs.role == "UGV" and obs.active_candidate is not None:
            centre = getattr(obs.active_candidate, "region_center_xy", None)
            position = getattr(obs, "agent_position", None)
            gap = None
            if centre and position:
                try:
                    gap = math.hypot(float(position[0]) - float(centre[0]),
                                     float(position[1]) - float(centre[1]))
                except (TypeError, ValueError, IndexError):
                    gap = None
            if gap is not None:
                payload["candidate_distance_m"] = round(gap, 1)
                payload["in_confirm_range"] = bool(gap <= self.max_confirm_distance_m)
        return payload

    @staticmethod
    def _bounded(image):
        """Keep request images inside the frozen 768x480 SFT thumbnail bound.

        Training rows are decoded with that same bound, so an unchanged image is
        byte-identical at train and inference time; larger frames are LANCZOS
        thumbnailed here instead of being re-sampled differently by the trainer.
        """
        if not image:
            return image
        try:
            with Image.open(io.BytesIO(image)) as source:
                frame = source.convert("RGB")
                if frame.width <= 768 and frame.height <= 480:
                    return image
                frame.thumbnail((768, 480), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                frame.save(output, format="PNG")
                return output.getvalue()
        except Exception:
            return image

    def _images(self, obs, proposals):
        scene = (render_legacy_sixview_collage(obs.ugv_views) if obs.role == "UGV" and obs.ugv_views
                 else getattr(obs, "raw_agent_image", obs.agent_image))
        return [self._bounded(image) for image in
                (obs.target_reference_image, scene, proposal_board(proposals)) if image]

    def _capture_dir(self, obs):
        """Per-scene capture directory: episode ids repeat across towns.

        The target reference image is fixed for one scene, so its digest keeps
        two towns' ``lite_easy_00002`` captures from overwriting each other.
        """
        directory = Path(self.capture_root) / str(obs.episode_id)
        reference = getattr(obs, "target_reference_image", None)
        if reference:
            return directory.with_name(directory.name + "-" +
                                       hashlib.sha256(reference).hexdigest()[:10])
        return directory

    def _request(self, obs, payload, images, call_kind):
        if self._remaining <= 0:
            return {"status": "budget_exhausted"}
        self._remaining -= 1
        self._calls += 1
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        event = {"episode_id": obs.episode_id, "role": obs.role, "step": obs.step,
                 "call_id": f"{obs.episode_id}:{obs.role}:{obs.step}:{len(self.events)}",
                 "call_kind": call_kind, "public_input": payload,
                 "system_prompt": _system_for(call_kind, self.verify_system_v2), "user_prompt": prompt,
                 "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                 # Raw pixels are never archived: bytes in the event would make
                 # ``ag_os_trace.jsonl`` unserializable and kill the whole
                 # evidence write (2026-10-28 smoke: every trace 0 bytes).
                 "image_count": len(images),
                 "image_sha256": [hashlib.sha256(image).hexdigest() for image in images],
                 "status": "started"}
        if self.capture_root:
            directory = self._capture_dir(obs)
            directory.mkdir(parents=True, exist_ok=True)
            stem = event["call_id"].replace(":", "_").replace("/", "_")
            paths = []
            for index, image in enumerate(images):
                path = directory / ("%s_%02d.png" % (stem, index))
                path.write_bytes(image)
                paths.append(str(path))
            event["image_paths"] = paths
        self.events.append(event)
        try:
            _split_client = (self._vlm_ugv if obs.role == "UGV"
                             else self._vlm_uav)
            parsed, status, raw = _split_client.decide(
                _system_for(call_kind, self.verify_system_v2), prompt,
                [base64.b64encode(image).decode("ascii") for image in images])
            self.last_parse_status, self.last_raw_response = status, raw
            event.update(status=status, raw_response=raw)
            if status.startswith("vlm_"):
                raise RuntimeError("COOP_MODEL_INFRA: " + status)
            if not isinstance(parsed, dict):
                event["status"] = "invalid_response"
                return {}
            return parsed
        except Exception as error:
            event.update(status="error", error=f"{type(error).__name__}: {error}")
            raise

    def decide(self, obs, state: dict, proposals: list, call_reason: str,
               remaining_calls: int) -> dict:
        self._remaining = max(0, int(remaining_calls))
        if not call_reason:
            return {}
        if self._remaining <= 0:
            # L1: distinguish "the framework asked with an empty budget" from a
            # model fault; the method layer labels this decision_budget_exhausted.
            return {"status": "budget_exhausted"}
        payload = self._input(obs, state, proposals, call_reason, "primary_decision")
        parsed = self._request(obs, payload, self._images(obs, proposals), "primary_decision")
        decision = parsed.get("decision", parsed)
        result = {key: deepcopy(decision[key]) for key in
                  ("owner_choice", "request_reply", "request_id", "request_assistance",
                   "sender_policy", "resume_search")
                  if key in decision}
        action = parsed.get("action", {})
        if isinstance(action, str):
            action = parsed
        if not isinstance(action, dict):
            action = {}
        result["action"] = {key: deepcopy(action[key]) for key in
                            ("role", "action", "waypoint_id", "candidate_id", "target_verdict",
                             "evidence_view", "proposal_id", "fallback_action", "fallback_waypoint_id")
                            if key in action}
        self.events[-1]["target"] = deepcopy(result)
        return result

    def _write_site(self, obs, payload, images, call_kind, call_id, target, capture_site):
        """Persist one decision site (prompt bytes, images, label) without a call."""
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        event = {"episode_id": obs.episode_id, "role": obs.role, "step": obs.step,
                 "call_id": call_id, "call_kind": call_kind, "public_input": payload,
                 "system_prompt": _system_for(payload["call_kind"], self.verify_system_v2), "user_prompt": prompt,
                 "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                 "image_sha256": [hashlib.sha256(image).hexdigest() for image in images],
                 "status": "expert", "target": deepcopy(target),
                 "capture_site": capture_site}
        if self.capture_root:
            directory = self._capture_dir(obs)
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for index, image in enumerate(images):
                path = directory / ("%s_%02d.png" % (call_id.replace(":", "_"), index))
                path.write_bytes(image)
                paths.append(str(path))
            event["image_paths"] = paths
        self.events.append(event)
        return event

    def record_expert(self, obs, state: dict, proposals: list, call_reason: str, target: dict) -> dict:
        """Record a GT-expert decision at a model site; costs zero real calls.

        The prompt carries the *deployment* call identity so a frozen row sees the
        same input surface the student sees; ``call_kind`` in the archived event
        keeps the provenance that this label came from the expert.
        """
        payload = self._input(obs, state, proposals, call_reason, "primary_decision")
        return self._write_site(obs, payload, self._images(obs, proposals),
                                "expert_decision",
                                f"{obs.episode_id}:{obs.role}:{obs.step}:expert",
                                target, "expert_decision")

    def record_expert_verify(self, obs, proposals: list, label: dict, call_reason: str,
                             phase: str = "coverage") -> dict:
        """Record the deployment's second (crop-verification) site for an expert.

        The payload is identical to :meth:`verify_proposals` -- same public input,
        same ``required_response``, same images -- so a frozen row matches what the
        student is asked at deployment. Only the label source differs, and that is
        recorded in ``capture_site``.
        """
        payload = self._input(obs, {"phase": phase}, proposals,
                              "candidate_verify", "gsam_crop")
        payload["required_response"] = {
            "selected_proposal": "shown proposal ID or NONE",
            "target_verdict": "MATCH|NON_MATCH|UNKNOWN"}
        return self._write_site(obs, payload, self._images(obs, proposals), "expert_verify",
                                f"{obs.episode_id}:{obs.role}:{obs.step}:expert_verify",
                                label, "expert_verify")

    def _confirm_images(self, obs):
        """Ground-resolution evidence only: the reference plus the six around-views.

        v13 E2: the confirm site is the only place identity is decided, so it
        ships the native-resolution views (six-view collage 1440x540) instead of
        the frozen 768x480 SFT thumbnail bound; the 2026-11-06 audit found the
        bound shrank each panel to 256x96 while the served context (16384) had
        ample room. ``_bounded`` stays on every other call site.
        """
        scene = render_legacy_sixview_collage(obs.ugv_views) if obs.ugv_views else None
        return [image for image in (obs.target_reference_image, scene) if image]

    def record_expert_confirm(self, obs, label: dict, call_reason: str) -> dict:
        """Record the ground confirmation site for an expert, with its own label.

        Deployment asks ``verify_confirm`` when the UGV stands at the candidate;
        the expert must label that exact site or the adapter is trained on a
        question it will never be asked. Payload and images come from the same
        code path ``verify_confirm`` uses, so a frozen row sees the deployment
        bytes (including ``candidate_distance_m``/``in_confirm_range``).

        Only the two-key answer becomes the target: ``validate_target`` rejects
        any unknown target key, and an ``observable_support`` block inside the
        label silently made every captured row ineligible (2026-11-01 dry run:
        8 confirmation rows archived, 0 in the dataset). The support is kept on
        the event instead, where it audits the label without training on it.
        """
        payload = self._input(obs, {"phase": "verify"}, [], "confirm_verify", "confirm_verify")
        payload["required_response"] = {
            "target_verdict": "MATCH|NON_MATCH|UNKNOWN",
            "rule": "MATCH only when the confirmed vehicle is the reference vehicle"}
        target = {"target_verdict": label.get("target_verdict")}
        event = self._write_site(obs, payload, self._confirm_images(obs), "expert_confirm",
                                 f"{obs.episode_id}:{obs.role}:{obs.step}:expert_confirm",
                                 target, "expert_confirm")
        event["expert_support"] = deepcopy(label.get("observable_support") or {})
        return event

    def verify_confirm(self, obs) -> dict:
        """Ask the narrow identity question at the ground confirm site, once."""
        if self._remaining <= 0:
            return {"status": "budget_exhausted"}
        payload = self._input(obs, {"phase": "verify"}, [], "confirm_verify", "confirm_verify")
        payload["required_response"] = {
            "target_verdict": "MATCH|NON_MATCH|UNKNOWN",
            "rule": "MATCH only when the confirmed vehicle is the reference vehicle"}
        parsed = self._request(obs, payload, self._confirm_images(obs), "confirm_verify")
        result = _verify_answer(parsed)
        verdict = result.get("target_verdict")
        final = {"status": "ok",
                 "target_verdict": verdict if verdict in ("MATCH", "NON_MATCH") else "UNKNOWN",
                 "raw": deepcopy(result)}
        self.events[-1]["target"] = deepcopy(final)
        return final

    def verify_proposals(self, obs, proposals: list, phase: str = "coverage") -> dict:
        if self._remaining <= 0:
            return {"status": "budget_exhausted"}
        payload = self._input(obs, {"phase": phase}, proposals, "candidate_verify", "gsam_crop")
        payload["required_response"] = {
            "selected_proposal": "shown proposal ID or NONE", "target_verdict": "MATCH|NON_MATCH|UNKNOWN"}
        parsed = self._request(obs, payload, self._images(obs, proposals), "gsam_crop")
        result = _verify_answer(parsed)
        raw = deepcopy(result) if result else {}
        valid_ids = {_value(proposal, "proposal_id") for proposal in proposals}
        verdict = result.get("target_verdict")
        selected = result.get("selected_proposal")
        # M2': a MATCH must be bound to a real shown proposal; MATCH+NONE and
        # hallucinated IDs are coerced to the unknown family instead of reaching
        # the menu lookup as proposal_not_in_menu.
        if (selected not in valid_ids) or (verdict == "MATCH" and selected == "NONE"):
            result = {"selected_proposal": "NONE",
                      "target_verdict": verdict if verdict in ("NON_MATCH", "UNKNOWN") else "UNKNOWN"}
        result["status"] = "ok"
        self.events[-1]["verify_raw"] = raw
        self.events[-1]["target"] = deepcopy(result)
        return result

    def drain_calls(self) -> int:
        result, self._calls = self._calls, 0
        return result

    def drain_events(self) -> list:
        events, self.events = self.events, []
        return events
