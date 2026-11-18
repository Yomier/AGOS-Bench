# -*- coding: utf-8 -*-
"""legacy 2x3 six-view collage adapter（r7 边界重构）。

2x3 collage 是 需要多视图拼图时使用的方法内 视觉表示，
不再属于 benchmark。本模块严格复现 carla_interface.capture_ugv_6view
的历史渲染格式，保证旧方法视觉语义不变：

- 1440x540（panel 480x270）2 行 3 列
- 上行 FRONT_LEFT / FRONT / FRONT_RIGHT，下行 BACK_LEFT / BACK / BACK_RIGHT
- 黄色 2px 边框 + DejaVuSans-Bold 16 黄色标签（缺字体回退 default）
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agos.lite_core import DirectionalView

_LEGACY_LABELS = ("FRONT_LEFT", "FRONT", "FRONT_RIGHT",
                  "BACK_LEFT", "BACK", "BACK_RIGHT")


def _load_font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def render_legacy_sixview_collage(ugv_views, panel_w=480, panel_h=270):
    """把六个 DirectionalView 渲染成 legacy 2x3 collage PNG bytes。

    与 carla_interface.capture_ugv_6view 的历史输出逐像素一致：
    - 有效视图：解码 PNG 后 resize 到 panel 尺寸
    - 黑帧（valid=False）：零图（其 image_bytes 本身即零图 PNG）
    - 少于 6 个视图时：缺失位补零图（防御，不改变正常 6 视图行为）
    """
    from collections.abc import Iterable
    if not isinstance(ugv_views, Iterable):
        raise TypeError("ugv_views must be iterable of DirectionalView")
    views = list(ugv_views)
    n = len(views)
    if n > 6:
        raise ValueError("legacy collage expects at most 6 views, got %d" % n)

    collage = Image.new("RGB", (panel_w * 3, panel_h * 2), (0, 0, 0))
    draw = ImageDraw.Draw(collage)
    font = _load_font()

    for i in range(6):
        if i < n:
            v = views[i]
            try:
                pil = Image.open(io.BytesIO(v.image_bytes)).convert("RGB")
            except Exception:
                pil = Image.fromarray(
                    np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
        else:
            pil = Image.fromarray(
                np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
        pil = pil.resize((panel_w, panel_h))
        row, col = i // 3, i % 3
        x_off = col * panel_w
        y_off = row * panel_h
        collage.paste(pil, (x_off, y_off))
        draw.rectangle([x_off, y_off, x_off + panel_w - 1, y_off + panel_h - 1],
                       outline=(255, 255, 0), width=2)
        draw.text((x_off + 5, y_off + 3), _LEGACY_LABELS[i],
                  fill=(255, 255, 0), font=font)

    buf = io.BytesIO()
    collage.save(buf, format="PNG")
    return buf.getvalue()
