from __future__ import annotations

import hashlib

import io

from typing import Optional, Tuple

def _decode_png(b64_str: str):
    if not b64_str:
        return None
    try:
        import base64
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception:
        return None

class _SourceCache:
    """单槽源图缓存：同一帧的多个 proposal 不重复解码源 PNG。"""

    def __init__(self):
        self._key = None
        self._img = None

    def get(self, source_bytes: bytes):
        key = hashlib.sha256(source_bytes or b"").hexdigest()
        if key != self._key:
            try:
                from PIL import Image
                self._img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
            except Exception:
                self._img = None
            self._key = key
        return self._img

_CACHE = _SourceCache()

def server_crop_box(bbox_xyxy, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    """sidecar /v1/gsam 的实际裁剪坐标（与 gsam_sidecar_server 同规则）。"""
    bb = list(bbox_xyxy or [])
    x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
    x2, y2 = min(int(image_w), int(bb[2]) + 1), min(int(image_h), int(bb[3]) + 1)
    return x1, y1, x2, y2

def verify_proposal_source(source_bytes: bytes, proposal) -> Tuple[bool, str]:
    """像素级校验 proposal 的 raw crop 确实来自该源图同 bbox 裁剪。

    返回 (ok, reason)；ok=False 时该 proposal 必须被拒绝（source 错误分流，
    不能与空检测混同）。解码失败按不匹配处理。
    """
    crop = _decode_png(getattr(proposal, "crop_b64", "") or "")
    if crop is None:
        return False, "crop_undecodable"
    src = _CACHE.get(source_bytes or b"")
    if src is None:
        return False, "source_undecodable"
    box = server_crop_box(getattr(proposal, "bbox_xyxy", []), src.size[0], src.size[1])
    expect = src.crop(box)
    if expect.size[0] < 2 or expect.size[1] < 2:
        return False, "crop_too_small"
    if expect.size != crop.size:
        return False, "crop_size_mismatch"
    if expect.tobytes() != crop.tobytes():
        return False, "pixel_mismatch"
    return True, "ok"
