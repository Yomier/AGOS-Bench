from __future__ import annotations

import base64

import hashlib

import json

import math

import time

from dataclasses import dataclass, field

from typing import Any, Callable, Dict, List, Optional, Tuple


VIEW_ORDER: Tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4", "V5")

VIEW_LABELS: Dict[str, str] = {
    "V0": "FRONT_LEFT (-45°)",
    "V1": "FRONT (0°)",
    "V2": "FRONT_RIGHT (+45°)",
    "V3": "BACK_LEFT (-135°)",
    "V4": "BACK (180°)",
    "V5": "BACK_RIGHT (+135°)",
}

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b or b"").hexdigest()

@dataclass
class UgvEvidenceBundle:
    """一次 UGV 观测的六视图证据包（设计 §4.1）。

    六路顺序采集；绑定 observation_bundle_id；准备核验时复用该包，
    不重拍、不移动、不刷新候选板。
    """
    bundle_id: str
    episode_id: str
    step: int
    views: List[Dict[str, Any]] = field(default_factory=list)  # V0..V5 原始记录
    reference_sha256: str = ""
    # sidecar 证据板（ugv_grounded_evidence_enabled 时）
    evidence: Dict[str, Any] = field(default_factory=dict)

    def view_bytes(self, view_id: str) -> bytes:
        for v in self.views:
            if v.get("view_id") == view_id:
                return v.get("image_bytes") or b""
        return b""

    def view_hashes(self) -> Tuple[str, ...]:
        return tuple(v.get("sha256", "") for v in self.views)

    def shown_proposal_ids(self, view_id: Optional[str] = None) -> List[str]:
        ids: List[str] = []
        for p in (self.evidence.get("proposals") or []):
            if p.get("shown"):
                if view_id is None or p.get("view_id") == view_id:
                    ids.append(str(p.get("proposal_id")))
        return ids

    def board_sha256(self) -> str:
        return str(self.evidence.get("board_sha256") or "")

def _default_post(endpoint: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    import requests
    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

class UgvGroundedEvidenceClient:
    """POST {sidecar}/v1/ugv_views 客户端。

    工具最多 6 panel 前向/轮、服务端顺序处理；deadline=120s、最多 1
    attempt（设计 §6）。异常状态分为 empty / partial_error / error /
    invalid_source；正常轮异常时主模型仍看原六图。
    """

    def __init__(self, endpoint: str, timeout_s: float = 120.0,
                 box_threshold: float = 0.20, text_threshold: float = 0.18,
                 nms_iou: float = 0.50, panel_limit: int = 8,
                 display_limit: int = 3, post: Callable = _default_post,
                 evidence_id_mode: str = "full"):
        # L12 教训（sidecar URL 旧契约 404 被错误分流吞掉）：endpoint 不含
        # 路径时自动补 /v1/ugv_views，杜绝裸 host 404。
        ep = str(endpoint or "").strip()
        if ep and "/" not in ep.split(":", 3)[-1]:
            ep = ep.rstrip("/") + "/v1/ugv_views"
        self.endpoint = ep
        self.timeout_s = float(timeout_s)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.panel_limit = int(panel_limit)
        self.display_limit = int(display_limit)
        self._post = post
        self.evidence_id_mode = evidence_id_mode

    def request_board(self, bundle: UgvEvidenceBundle,
                      reference_bytes: bytes,
                      detection_query: str = "car . van . truck . bus . vehicle .",
                      ) -> Dict[str, Any]:
        """请求证据板。返回 evidence dict（原样挂 bundle.evidence 与 obs）。"""
        invalid = invalid_capture_views(bundle)
        if invalid:
            return {"status": "invalid_frame", "error": "sensor_capture_exhausted",
                    "invalid_views": invalid, "proposals": [], "panels": [], "board_b64": ""}
        views = []
        for v in bundle.views:
            views.append({
                "view_id": str(v.get("view_id")),
                "image_b64": base64.b64encode(v.get("image_bytes") or b"").decode("ascii"),
                "source_sha256": str(v.get("sha256", "")),
                "width": int(v.get("width", 480) or 480),
                "height": int(v.get("height", 270) or 270),
                "valid": bool(v.get("valid", True)),
            })
        payload = {
            "domain": "ugv",
            "bundle_id": bundle.bundle_id,
            "views": views,
            "target_reference_b64": base64.b64encode(reference_bytes or b"").decode("ascii"),
            "detection_query": detection_query,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_iou": self.nms_iou,
            "panel_limit": self.panel_limit,
            "display_limit": self.display_limit,
            "segment": False,  # 设计 v2：正常轮与核验均固定 segment=false
            "evidence_id_mode": self.evidence_id_mode,
        }
        try:
            obj = self._post(self.endpoint, payload, self.timeout_s)
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}",
                    "proposals": [], "panels": [], "board_b64": ""}
        status = str(obj.get("status") or "")
        if status not in ("ok", "empty", "partial_error", "error"):
            return {"status": "error", "error": f"invalid_sidecar_status:{status}",
                    "proposals": [], "panels": [], "board_b64": ""}
        # Transport failures may fall back to raw RGB; misbound evidence is a
        # distinct contract failure and must never be attached to this position.
        if status != "error" or obj.get('panels'):
            expected = {v["view_id"]: v["sha256"] for v in bundle.views}
            panels = obj.get("panels") or []
            actual = {p.get("view_id"): p.get("source_sha256") for p in panels if isinstance(p, dict)}
            if (obj.get("bundle_id") != bundle.bundle_id or len(panels) != 6
                    or actual != expected):
                return {"status": "invalid_source", "error": "response_bundle_or_source_mismatch",
                        "proposals": [], "panels": panels, "board_b64": ""}
            invalid = [p for p in panels if p.get('status') == 'invalid_source']
            if invalid:
                return {"status": "invalid_source", "error": "tool_panel_invalid_source:" + json.dumps(invalid),
                        "proposals": [], "panels": panels, "board_b64": ""}
        return obj

def invalid_capture_views(bundle):
    """Missing sensor frames are acquisition failures, never valid tool inputs."""
    return [str(v.get('view_id')) for v in bundle.views
            if not v.get('valid', True) or not v.get('image_bytes')]

def build_evidence_bundle(obs) -> UgvEvidenceBundle:
    """从 obs.ugv_views 构建 UgvEvidenceBundle（不重拍）。"""
    views = []
    source_views = list(getattr(obs, "ugv_views", None) or ())
    source_ids = [str(getattr(v, "view_id", "")) for v in source_views]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) - set(VIEW_ORDER):
        raise ValueError("sixview_input_contract: duplicate or unexpected source panel")
    by_id = dict(zip(source_ids, source_views))
    for i, vid in enumerate(VIEW_ORDER):
        v = by_id.get(vid)
        if v is None:
            views.append({"view_id": vid, "image_bytes": b"", "sha256": "",
                          "width": 0, "height": 0, "valid": False,
                          "frame_id": None, "view_name": VIEW_LABELS[vid].split(" ")[0],
                          "yaw_offset_deg": _REGISTRY_YAW.get(vid)})
            continue
        data = getattr(v, "image_bytes", b"") or b""
        views.append({
            "view_id": vid,
            "view_name": str(getattr(v, "view_name", "") or ""),
            "yaw_offset_deg": float(getattr(v, "yaw_offset_deg", 0.0) or 0.0),
            "width": int(getattr(v, "width", 480) or 480),
            "height": int(getattr(v, "height", 270) or 270),
            "frame_id": getattr(v, "frame_id", None),
            "timestamp_s": getattr(v, "timestamp_s", None),
            "valid": bool(getattr(v, "valid", True)),
            "image_bytes": data,
            "sha256": _sha256(data),
        })
    bundle = UgvEvidenceBundle(
        # 短 bundle id（8 位哈希前缀）：proposal ID 形如 Bxxxxxxx/Vn/Pm，
        # 可被 MLLM 在 JSON 中可靠复述（开发集教训：sample 全名做 ID 时
        # 模型抄不对，触发合同全判协议非法）。episode+step 仍保证唯一。
        bundle_id="B%s" % _sha256(("%s|ugv|%d" % (getattr(obs, "episode_id", "ep"),
                                                  int(getattr(obs, "step", 0) or 0)))
                                  .encode("utf-8"))[:8],
        episode_id=str(getattr(obs, "episode_id", "")),
        step=int(getattr(obs, "step", 0) or 0),
        views=views,
        reference_sha256=_sha256(getattr(obs, "target_reference_image", b"") or b""),
    )
    return bundle

_REGISTRY_YAW = {"V0": -45.0, "V1": 0.0, "V2": 45.0, "V3": -135.0, "V4": 180.0, "V5": 135.0}
