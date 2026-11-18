from __future__ import annotations

import base64

import hashlib

import io

import math

import time

from dataclasses import dataclass, field

from typing import Any, Callable, Dict, List, Optional, Tuple

MIN_MASK_AREA_PX = 24

MIN_MASK_BBOX_SIDE_PX = 4

MIN_MASK_BOX_IOU = 0.10          # mask 紧包围框 与 提示检测框 的 IoU 下限

MAX_MASK_IMAGE_AREA_RATIO = 0.95  # mask 面积占全图比例上限（异常整图连通）

@dataclass
class GroundedProposal:
    proposal_id: str                    # "P1".."Pk"，按 combined score 降序稳定编号
    bbox_xyxy: List[float]              # 原始像素检测框（缩放回原图后）
    detector_score: float = 0.0
    identity_score: float = 0.0         # SigLIP2 normalized (cosine+1)/2 from service, not raw cosine
    mask_bbox_xyxy: Optional[List[float]] = None   # mask 紧包围框（无 mask 时 None）
    mask_area_px: int = 0
    crop_b64: str = ""                  # 原始 RGB 局部裁剪（PNG b64）
    masked_crop_b64: str = ""           # 去背景分割图（PNG b64；仅 segment=true）
    source: str = "gsam_sidecar"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d

@dataclass
class GroundResult:
    status: str                         # ok / empty / fallback_no_crop / error / unavailable / invalid_frame / invalid_source
    proposals: List[GroundedProposal] = field(default_factory=list)
    frame_id: Optional[int] = None
    source_image_sha256: str = ""
    latency_ms: float = 0.0
    error: str = ""
    sam_model: str = ""
    backend: str = "gsam_sidecar"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "frame_id": self.frame_id,
            "source_image_sha256": self.source_image_sha256,
            "latency_ms": round(self.latency_ms, 1), "error": self.error,
            "sam_model": self.sam_model, "backend": self.backend,
            "proposals": [p.to_dict() for p in self.proposals],
        }

def _default_post(endpoint: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    import requests
    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b or b"").hexdigest()

def mask_bbox_from_mask_quality(mask_bbox, mask_area_px, image_w: int, image_h: int,
                                prompt_bbox: List[float]) -> Tuple[bool, str]:
    """mask 合法性检查（设计 §4.2）：返回 (valid, reason)。"""
    if not mask_bbox or len(mask_bbox) != 4:
        return False, "invalid_mask_bbox"
    x1, y1, x2, y2 = map(float, mask_bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, *prompt_bbox)):
        return False, "nonfinite_mask_geometry"
    if x2 <= x1 or y2 <= y1:
        return False, "degenerate_mask_bbox"
    if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
        return False, "mask_bbox_out_of_image"
    if int(mask_area_px) < MIN_MASK_AREA_PX:
        return False, "mask_area_too_small"
    if (x2 - x1) < MIN_MASK_BBOX_SIDE_PX or (y2 - y1) < MIN_MASK_BBOX_SIDE_PX:
        return False, "mask_bbox_too_thin"
    if int(mask_area_px) > MAX_MASK_IMAGE_AREA_RATIO * image_w * image_h:
        return False, "mask_covers_image"
    # 与提示框的 IoU
    px1, py1, px2, py2 = map(float, prompt_bbox)
    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (x2 - x1) * (y2 - y1)
    a2 = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    if a2 <= 0:
        return False, "invalid_prompt_bbox"
    if inter / max(a1, a2) < MIN_MASK_BOX_IOU:
        return False, "mask_box_iou_too_low"
    return True, "ok"

class GroundedLocalizer:
    """GSAM sidecar 客户端。frame 采用 duck-typing（lite_core.UAVFrame 兼容）：
    需要 image_bytes/valid/frame_id/x/y/z/yaw_deg/fov_deg/width/height。"""

    def __init__(self, endpoint: str, timeout_s: float = 20.0,
                 box_threshold: float = 0.20, text_threshold: float = 0.18,
                 nms_iou: float = 0.50, max_proposals: int = 3,
                 post: Callable = _default_post):
        self.endpoint = str(endpoint or "").strip()
        self.timeout_s = float(timeout_s)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.max_proposals = int(max_proposals)
        self._post = post

    # ---------------- frame 契约 ----------------
    @staticmethod
    def _frame_ok(frame) -> Tuple[bool, str]:
        if frame is None:
            return False, "missing_frame"
        if not bool(getattr(frame, "valid", True)):
            return False, "frame_invalid"
        if not getattr(frame, "image_bytes", None):
            return False, "frame_has_no_image"
        if int(getattr(frame, "width", 0) or 0) <= 0 or int(getattr(frame, "height", 0) or 0) <= 0:
            return False, "frame_size_invalid"
        return True, ""

    # ---------------- 主入口：ground ----------------
    def ground_frame(self, frame, target_description: str,
                     reference_bytes: bytes, *, segment: bool = True) -> GroundResult:
        t0 = time.time()
        ok, why = self._frame_ok(frame)
        sha = _sha256(getattr(frame, "image_bytes", b"") or b"")
        res = GroundResult(status="invalid_frame", frame_id=getattr(frame, "frame_id", None),
                           source_image_sha256=sha, error=why)
        if not ok:
            return res
        if not self.endpoint:
            res.status, res.error = "unavailable", "missing_endpoint"
            return res

        payload = {
            "image_b64": base64.b64encode(frame.image_bytes).decode("ascii"),
            # 随机单图整张传输（服务端整图编码；无三视图切分字段）
            "target_reference_b64": base64.b64encode(reference_bytes or b"").decode("ascii"),
            "reference_mode": "random_single",
            "target_description": str(target_description or "vehicle"),
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_iou": self.nms_iou,
            "topk": self.max_proposals,
            "segment": bool(segment),
        }
        try:
            obj = self._post(self.endpoint, payload, self.timeout_s)
        except Exception as e:  # 服务错误单列，绝不伪装成"没有目标"
            res.latency_ms = (time.time() - t0) * 1000.0
            res.status, res.error = "error", f"{type(e).__name__}: {e}"
            return res
        res.latency_ms = (time.time() - t0) * 1000.0
        if not isinstance(obj, dict) or not isinstance(obj.get("proposals", []), list):
            res.status, res.error = "error", "invalid_tool_response_schema"
            return res
        res.sam_model = str(obj.get("sam_model") or "")

        status = str(obj.get("status") or "")
        if status == "error":
            res.status, res.error = "error", str(obj.get("error") or "server_error")
            return res

        w, h = int(frame.width), int(frame.height)
        props_in = list(obj.get("proposals") or [])
        if not props_in:
            res.status = "empty"          # 空检测：可回退（无裁剪路径）
            return res

        out: List[GroundedProposal] = []
        seg_failed = 0
        for i, p in enumerate(props_in[: self.max_proposals]):
            if not isinstance(p, dict):
                continue
            try:
                bb = [float(v) for v in (p.get("bbox_xyxy") or [])]
                detector_score = float(p.get("detector_score") or 0.0)
                identity_score = float(p.get("identity_score") or 0.0)
                mask_area = int(p.get("mask_area_px") or 0)
                mb = p.get("mask_bbox_xyxy")
                if mb is not None:
                    mb = [float(v) for v in mb]
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in (*bb, detector_score, identity_score)):
                continue
            if len(bb) != 4 or bb[2] <= bb[0] or bb[3] <= bb[1]:
                continue                  # 框合法性：非法框直接丢弃
            if bb[0] < -1 or bb[1] < -1 or bb[2] > w + 1 or bb[3] > h + 1:
                continue
            if segment:
                if mb is None:
                    seg_failed += 1
                    continue              # segment 模式无有效 mask → 该 proposal 不可用
                valid, reason = mask_bbox_from_mask_quality(
                    mb, mask_area, w, h, bb)
                if not valid:
                    seg_failed += 1
                    continue              # 部分 mask 失败只剔除该 proposal
                mb = [float(v) for v in mb]
            else:
                mb = None                 # 实验二组件：无 SAM，无 mask 字段
            raw_crop = str(p.get("crop_b64") or "")
            masked_crop = str(p.get("masked_crop_b64") or "") if segment else ""
            if segment and decode_png_b64(masked_crop) is None:
                res.status, res.error = "invalid_source", f"proposal_{i + 1}: masked_crop_undecodable"
                return res
            proposal = GroundedProposal(
                proposal_id=f"P{len(out) + 1}",
                bbox_xyxy=bb,
                detector_score=detector_score,
                identity_score=identity_score,
                mask_bbox_xyxy=mb,
                mask_area_px=mask_area,
                crop_b64=raw_crop,
                masked_crop_b64=masked_crop,
                meta={"server_rank": i},
            )
            # The service does not echo a trusted input hash. Re-crop the
            # actual source frame using the same pixel contract as fixed arms.
            from agos.methods.common.source_guard import verify_proposal_source
            source_ok, source_error = verify_proposal_source(frame.image_bytes, proposal)
            if not source_ok:
                res.status, res.error = "invalid_source", f"proposal_{i + 1}: {source_error}"
                return res
            out.append(proposal)
        res.proposals = out
        if out:
            res.status = "ok"
        else:
            # 有检测但全部因 mask 失败被剔除：无有效裁剪 → 直接网格回退路径
            res.status = "fallback_no_crop"
            res.error = f"all_{len(props_in)}_proposals_failed_mask_check" if seg_failed else "no_valid_proposal"
        return res

    # ---------------- 投影：point 定位锚 ----------------
    def project_proposal(self, frame, proposal: GroundedProposal,
                         ground_z: Optional[float] = None, *,
                         ground_elevation=None) -> Optional[Tuple[float, float]]:
        """把 proposal 的定位锚（mask 紧包围框中心；无 mask 时用检测框中心）
        投影到世界坐标 (x, y)。使用源帧传感器位姿与唯一相机几何真源。"""
        ok, _ = self._frame_ok(frame)
        if not ok or proposal is None:
            return None
        w, h = int(frame.width), int(frame.height)
        anchor = proposal.mask_bbox_xyxy if proposal.mask_bbox_xyxy else proposal.bbox_xyxy
        x1, y1, x2, y2 = map(float, anchor)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return None
        # 中心小方形（±2px）投影取质心，保持与几何真源同一条代码路径
        s = 2.0
        poly_px = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
        try:
            from agos.perception.uav_grid_geometry import pixel_polygon_to_world_polygon
            args = (float(frame.x), float(frame.y), float(frame.z),
                float(getattr(frame, "yaw_deg", 0.0) or 0.0),
                float(getattr(frame, "fov_deg", 90.0) or 90.0),
                w, h, poly_px)
            plane = float(ground_z) if ground_z is not None else 0.
            _, (wx, wy), _ = pixel_polygon_to_world_polygon(*args, ground_z=plane)
            # One public-road approximation: locate on z=0, sample that cell,
            # and reproject the SAME source-pixel center once. No actor/GT z.
            if ground_z is None and ground_elevation is not None:
                try:
                    value = ground_elevation(wx, wy)
                    height = float(value) if value is not None else 0.
                    if not math.isfinite(height):
                        height = 0.
                except Exception:
                    height = 0.
                if height != 0.:
                    _, (wx, wy), _ = pixel_polygon_to_world_polygon(*args, ground_z=height)
            return float(wx), float(wy)
        except Exception:
            return None

def project_observation_proposal(localizer, obs, frame, proposal):
    """Forward only enabled PUBLIC height data; legacy localizers keep two args."""
    provider = getattr(obs, 'public_ground_elevation', None)
    if getattr(obs, 'candidate_use_ground_elevation', False) and provider is not None:
        return localizer.project_proposal(frame, proposal, ground_elevation=provider)
    return localizer.project_proposal(frame, proposal)

def decode_png_b64(b64_str: str):
    """辅助：解码 PNG b64 → PIL Image（无 PIL 环境返回 None）。"""
    if not b64_str:
        return None
    try:
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception:
        return None
