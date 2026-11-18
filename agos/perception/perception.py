# -*- coding: utf-8 -*-
"""perception.py — Grid overlay + Grid projector + Candidate ledger + Visibility checker + TopDown renderer"""
import math, io, json, logging
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# P0-1 fix: Use unified geometry module for all UAV grid mapping
try:
    from .uav_grid_geometry import (
        cell_label_to_world_polygon as _cell_to_world,
        world_to_grid_cell as _world_to_cell,
        load_calibration as _load_calib,
    )
    _load_calib()
except ImportError:
    _cell_to_world = None
    _world_to_cell = None

logger = logging.getLogger("LiteV1.Perception")

# B19（2026-09-17 审计）：render_topdown 覆盖边界内缩余量（原硬编码 ±20 提为模块常量统一）。
# 注意：与 maps.waypoints.MapFrame 的历史死参数 margin 无关（该参数因 benchmark 旧脚本
# 仍传入而保留，见 waypoints.py 注释）。
TOPDOWN_RENDER_MARGIN_M = 20

# ============================================================
# UAV Grid Overlay
# ============================================================
def draw_grid_overlay(image_bytes, rows=5, cols=5):
    """在 UAV 俯视图上叠加 5x5 grid + A1-E5 标签"""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    cell_w = w / cols
    cell_h = h / rows
    labels = ["A", "B", "C", "D", "E"]

    for r in range(rows + 1):
        y = int(r * cell_h)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 80), width=1)
    for c in range(cols + 1):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 80), width=1)

    for r in range(rows):
        for c in range(cols):
            label = f"{labels[r]}{c+1}"
            x = int(c * cell_w + 3)
            y = int(r * cell_h + 2)
            draw.text((x, y), label, fill=(255, 255, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def get_cell_corners_px(cell_label, img_w=1024, img_h=576, rows=5, cols=5):
    """获取 grid cell 的四个像素角点"""
    labels = ["A", "B", "C", "D", "E"]
    import re as _re
    if not _re.match(r"^[A-E][1-5]$", cell_label):
        raise ValueError(f"Invalid grid cell: {cell_label}")
    row = labels.index(cell_label[0])
    col = int(cell_label[1:]) - 1
    cell_w = img_w / cols
    cell_h = img_h / rows
    x1 = col * cell_w
    y1 = row * cell_h
    x2 = (col + 1) * cell_w
    y2 = (row + 1) * cell_h
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

# ============================================================
# Grid Projector (pixel → world)
# ============================================================
def project_cell_to_world(uav_x, uav_y, uav_z, yaw_deg, pitch_deg, fov_deg,
                          img_w, img_h, cell_corners_px, ground_z=0.0):
    """将 grid cell 像素角点反投影为世界坐标 polygon。

    2026-08-02 修复（bug review）：本函数曾自带一份"yaw 旋转公式"私有回退
    （px→+X, py→+Y，与现行相机契约旋转 90°），在标定未加载时会静默走错。
    P0-1 的"所有网格映射必须走 uav_grid_geometry"原则现在真正生效：
    本函数只做委托，无任何私有几何实现。
    B9（2026-09-17 审计）：ground_z 此前声明了却没传给下游——现已打通传递链。
    ground_z 为 cell 中心路面高程（地图信息，非目标 GT），默认 0.0=旧行为；
    非零时地面足迹按有效高度（uav_z - ground_z）计算。
    """
    from .uav_grid_geometry import pixel_polygon_to_world_polygon
    return pixel_polygon_to_world_polygon(uav_x, uav_y, uav_z, yaw_deg, fov_deg,
                                          img_w, img_h, cell_corners_px, ground_z=ground_z)


def compute_grid_cell_from_world(uav_x, uav_y, uav_z, yaw_deg, fov_deg,
                                  img_w, img_h, target_x, target_y, rows=5, cols=5):
    """反向计算：给定目标世界坐标，返回对应的 grid cell 标签 (A1-E5)

    2026-08-02 修复（bug review）：删除私有 yaw 回退，统一委托
    uav_grid_geometry.world_to_grid_cell（north-up 契约唯一真源）。
    返回: (cell_label, pixel_x, pixel_y) 或 (None, None, None) 如果在视野外
    """
    from .uav_grid_geometry import world_to_grid_cell
    return world_to_grid_cell(uav_x, uav_y, uav_z, yaw_deg, fov_deg,
                              img_w, img_h, target_x, target_y, rows, cols)


def point_in_polygon(x, y, polygon):
    """判断点 (x, y) 是否在多边形内或边界上。polygon 为 [(x0,y0), (x1,y1), ...]。"""
    if not polygon:
        return False

    # Boundary check first: on edge counts as inside
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Cross product = 0 and within bounding box
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9:
            if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return True

    # Standard ray casting
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if edge crosses the horizontal ray to the right of point
        if ((y1 > y) != (y2 > y)):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_intersect >= x - 1e-9:
                inside = not inside
    return inside


# ============================================================
# Candidate Ledger
# ============================================================
def _polygons_overlap(poly_a, poly_b, eps=1e-6):
    """B7（2026-09-17 审计）：轴对齐多边形相交判定（双向角点包含测试的等价实现）。

    网格 cell 反投影出的世界多边形为轴对齐矩形，此时"任一方角点落在另一方
    内部"等价于包围盒严格重叠。共边/仅触边不算相交（eps 收缩）——同行邻列
    cell 共享一条边、中心距 ~32m，必须放行而不是误判为重叠。
    """
    if not poly_a or not poly_b:
        return False
    ax = [float(p[0]) for p in poly_a]
    ay = [float(p[1]) for p in poly_a]
    bx = [float(p[0]) for p in poly_b]
    by = [float(p[1]) for p in poly_b]
    return (min(ax) + eps < max(bx) and min(bx) + eps < max(ax)
            and min(ay) + eps < max(by) and min(by) + eps < max(ay))


class CandidateLedger:
    def __init__(self, suppression_radius=30, max_total=5,
                 rejected_cooldown_steps=None, rejected_max_reopens=0):
        self.candidates = []
        self.suppression_radius = suppression_radius
        # P1-1 (2026-09-04 external review, user-approved): max_total=None 表示不设
        # 候选总数上限（论文口径"候选数不设上限/不写 five annotations cap"）。
        self.max_total = max_total
        self._next_id = 1
        self.rejected_regions = []  # legacy/public compatibility: (center_x, center_y)
        # R19-R is opt-in.  None preserves the historical permanent suppression
        # byte-for-byte in decision semantics.  Events distinguish model REJECT
        # (cooldown/reopen eligible) from framework UNSERVICEABLE (permanent).
        self.rejected_cooldown_steps = (None if rejected_cooldown_steps is None
                                        else max(0, int(rejected_cooldown_steps)))
        self.rejected_max_reopens = max(0, int(rejected_max_reopens))
        self._suppression_events = []

    def create(self, grid_cell, polygon, center, uncertainty, confidence, evidence, step,
               single_active=True, shadow=False, max_shadow=None,
               suppression_mode="radius", min_shift_m=6.0):
        """创建候选。

        Args:
            single_active: 如果为 True，则已有 active candidate（unverified/assigned_to_ugv/visited）时拒绝创建。
            shadow: R16-Q。为 True 时创建 shadow 状态候选（不进入 UGV active 流，
                供异步队列暂存）；shadow 创建不受 single_active 限制，但受 max_shadow
                队列上限限制。空间抑制（suppression_radius）对所有候选原样生效，
                即 R16-Q 复用 P0 的唯一空间去重层，不叠加第二套。
            max_shadow: shadow 队列上限（None 表示不限制；由 runner 按 config 传入）。
            suppression_mode: B7（2026-09-17 审计）。"radius"=旧行为（中心距 <
                suppression_radius 即抑制）；"cell_aware"=仅当 ①新 cell 世界多边形与
                既有候选多边形相交 或 ②中心距 < min_shift_m 时才抑制——修复 R19
                "10/10 吞掉相邻行重定位"（相邻行中心距 18m、多边形不相交 → 放行）。
                rejected 区抑制逻辑不受此参数影响（保持旧 radius 口径）。
            min_shift_m: cell_aware 模式的中心距抑制阈值（默认 6.0，与
                create_revision_shadow 的 min_shift_m 统一来源，由 runner 按 config 传入）。
        """
        if self.max_total is not None and len(self.candidates) >= self.max_total:
            return None, "max_reached"

        if shadow:
            if max_shadow is not None and self.shadow_count() >= max_shadow:
                return None, "shadow_queue_full"
        elif single_active:
            active = self.get_active()
            if active is not None:
                return None, "active_candidate_exists"

        # Check suppression (existing candidates).  Under R19-R, rejected
        # candidates are governed by the cooldown event below instead of acting
        # as an accidental second permanent suppression layer.
        for c in self.candidates:
            if c.status in ("suppressed_duplicate",):
                continue
            if self.rejected_cooldown_steps is not None and c.status == "rejected":
                continue
            d = math.sqrt((c.region_center_xy[0]-center[0])**2 + (c.region_center_xy[1]-center[1])**2)
            if suppression_mode == "cell_aware":
                # B7：cell_aware 抑制——仅当多边形相交或中心距 < min_shift_m 才抑制；
                # 不相交且距离达标的一律放行（相邻行/邻列重定位不再被 30m 半径吞掉）
                if d < float(min_shift_m) or _polygons_overlap(polygon, c.region_polygon):
                    return None, "suppressed_duplicate"
            elif d < self.suppression_radius:
                return None, "suppressed_duplicate"

        _reopen_events = []
        if self.rejected_cooldown_steps is None:
            # Historical behavior: permanent REJECT/UNSERVICEABLE suppression.
            for rx, ry in self.rejected_regions:
                d = math.sqrt((rx-center[0])**2 + (ry-center[1])**2)
                if d < self.suppression_radius:
                    return None, "rejected_region_suppressed"
        else:
            # R19-R: only model REJECT is reopenable.  UNSERVICEABLE remains
            # permanent because it represents a framework reachability failure.
            for ev in self._suppression_events:
                rx, ry = ev["center"]
                d = math.sqrt((rx-center[0])**2 + (ry-center[1])**2)
                if d >= self.suppression_radius:
                    continue
                if ev.get("kind") != "reject":
                    return None, "unserviceable_region_suppressed"
                age = max(0, int(step) - int(ev.get("step", 0)))
                if age < self.rejected_cooldown_steps:
                    return None, "rejected_region_cooldown"
                if int(ev.get("reopens", 0)) >= self.rejected_max_reopens:
                    return None, "rejected_region_reopen_exhausted"
                _reopen_events.append(ev)

        cid = f"C{self._next_id:03d}"
        self._next_id += 1
        from agos.lite_core import CandidateInfo
        candidate = CandidateInfo(
            candidate_id=cid,
            status="shadow" if shadow else "unverified",
            grid_cell=grid_cell,
            region_center_xy=center,
            region_polygon=polygon,
            uncertainty_radius_m=uncertainty,
            confidence=confidence,
            source_agent="UAV",
            evidence=evidence,
            created_step=step,
        )
        self.candidates.append(candidate)
        for ev in _reopen_events:
            ev["reopens"] = int(ev.get("reopens", 0)) + 1
        if _reopen_events:
            candidate.reopened_rejected_region = True
            # R20 audit fix: remember the quota charges so runner can roll them
            # back if the candidate is discarded by serviceability precheck.
            candidate._r20_reopen_events_charged = list(_reopen_events)
        return candidate, "created" if not shadow else "shadow_created"

    def rollback_reopen_charge(self, candidate):
        """Undo R19-R reopen quota charges for a candidate that never entered service.

        R19 charged the reopen budget inside ``create`` before the runner's
        serviceability precheck.  If that precheck then deleted the candidate,
        the region silently lost a reopen.  This rollback is idempotent and is
        called only on pre-serviceability deletion.
        """
        events = list(getattr(candidate, "_r20_reopen_events_charged", []) or [])
        if not events:
            return 0
        n = 0
        for ev in events:
            ev["reopens"] = max(0, int(ev.get("reopens", 0)) - 1)
            n += 1
        candidate._r20_reopen_events_charged = []
        candidate.reopened_rejected_region = False
        return n

    def nearest_candidate(self, center, statuses=None):
        """Return (candidate, distance_m) nearest to ``center`` without oracle data."""
        allowed = set(statuses) if statuses is not None else None
        best = None
        best_d = float("inf")
        for c in self.candidates:
            if c.status == "suppressed_duplicate":
                continue
            if allowed is not None and c.status not in allowed:
                continue
            d = math.hypot(float(c.region_center_xy[0]) - float(center[0]),
                           float(c.region_center_xy[1]) - float(center[1]))
            if d < best_d:
                best, best_d = c, d
        return best, best_d

    def create_revision_shadow(self, parent_candidate_id, grid_cell, polygon, center,
                               uncertainty, confidence, evidence, step, *,
                               max_shadow=2, min_shift_m=6.0, max_shift_m=30.0,
                               max_per_parent=1, min_gap_steps=3):
        """R20-M: create a bounded alternate localization for one active parent.

        This intentionally bypasses the duplicate radius *only for the named
        parent candidate*.  It never mutates an assigned candidate or its route.
        The alternate hypothesis enters the normal Q shadow queue and therefore
        cannot preempt the current UGV verification.  No GT fields are read.
        """
        parent = next((c for c in self.candidates if c.candidate_id == parent_candidate_id), None)
        if parent is None:
            return None, "revision_parent_missing"
        if parent.status not in ("unverified", "assigned_to_ugv"):
            return None, "revision_parent_not_active"
        if max_per_parent <= 0:
            return None, "revision_disabled_by_budget"
        shift = math.hypot(float(parent.region_center_xy[0]) - float(center[0]),
                           float(parent.region_center_xy[1]) - float(center[1]))
        if shift < float(min_shift_m):
            return None, "revision_shift_too_small"
        if shift > float(max_shift_m):
            return None, "revision_shift_too_large"
        # B21（2026-09-17 审计）：sibling 预算只统计 parent 仍存活的 revision——
        # parent 已被删除（如可服务性打回后 remove）的孤儿 revision 不再占用
        # max_per_parent 名额，防止删除路径下预算被幽灵候选永久锁死。
        _live_ids = {c.candidate_id for c in self.candidates}
        siblings = [c for c in self.candidates
                    if getattr(c, "r20_revision_parent_id", None) == parent_candidate_id
                    and c.r20_revision_parent_id in _live_ids]
        if len(siblings) >= int(max_per_parent):
            return None, "revision_parent_budget_exhausted"
        if siblings:
            last_step = max(int(getattr(c, "created_step", 0) or 0) for c in siblings)
            if int(step) - last_step < int(min_gap_steps):
                return None, "revision_min_gap"
        if max_shadow is not None and self.shadow_count() >= int(max_shadow):
            return None, "shadow_queue_full"
        if self.max_total is not None and len(self.candidates) >= self.max_total:
            return None, "max_reached"

        # Protect against colliding with any *other* candidate.  Only the parent
        # is exempt from the ordinary spatial suppression rule.
        for c in self.candidates:
            if c.candidate_id == parent_candidate_id or c.status == "suppressed_duplicate":
                continue
            d = math.hypot(float(c.region_center_xy[0]) - float(center[0]),
                           float(c.region_center_xy[1]) - float(center[1]))
            if d < self.suppression_radius:
                return None, "revision_duplicate_other_candidate"

        cid = f"C{self._next_id:03d}"
        self._next_id += 1
        from agos.lite_core import CandidateInfo
        candidate = CandidateInfo(
            candidate_id=cid, status="shadow", grid_cell=grid_cell,
            region_center_xy=center, region_polygon=polygon,
            uncertainty_radius_m=uncertainty, confidence=confidence,
            source_agent="UAV", evidence=evidence, created_step=step)
        candidate.r20_revision_parent_id = parent_candidate_id
        candidate.r20_revision_shift_m = float(shift)
        candidate.r20_revision_kind = "duplicate_parent_relocalization"
        self.candidates.append(candidate)
        return candidate, "r20_revision_shadow_created"

    def shadow_candidates(self):
        """R16-Q：当前 shadow 队列（按创建顺序 = FIFO 序）。"""
        return [c for c in self.candidates if c.status == "shadow"]

    def shadow_count(self):
        return len(self.shadow_candidates())

    def promote_next_shadow(self):
        """R16-Q FIFO dispatch：无 active 候选时，把最老的 shadow 候选提升为 unverified。

        保持 UGV 单 active 硬不变量：get_active() 仍只认 unverified/assigned_to_ugv/
        visited，因此提升后的候选才进入 UGV 流程。返回被提升的候选（无则 None）。
        """
        if self.get_active() is not None:
            return None
        for c in self.candidates:
            if c.status == "shadow":
                c.status = "unverified"
                return c
        return None

    def promote_best_shadow(self, score_attr="r17_visual_rank_score"):
        """R17-V gated dispatch: promote highest-scored shadow candidate.

        This is intentionally separate from FIFO and is only called when the
        experiment config explicitly enables candidate_ranker_dispatch_enabled.
        Missing/invalid scores sort below valid scores; ties keep FIFO order.
        """
        if self.get_active() is not None:
            return None
        shadows=[c for c in self.candidates if c.status == "shadow"]
        if not shadows:
            return None
        def key(c):
            v=getattr(c, score_attr, None)
            valid=isinstance(v, (int, float))
            return (1 if valid else 0, float(v) if valid else float("-inf"), -int(getattr(c,"created_step",0) or 0))
        c=max(shadows, key=key)
        c.status="unverified"
        return c

    def defer_candidate(self, candidate_id, max_retry=1):
        """R16-D：非破坏性 stall defer。ASSIGNED -> DEFERRED（shadow 态），
        不产生 REJECT、不计 false reject。每个候选最多 retry 一次，超过则 dropped。
        返回 (ok, reason)。"""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                if c.status != "assigned_to_ugv":
                    return False, "not_assigned"
                retries = int(getattr(c, "defer_retry_count", 0) or 0)
                c.defer_retry_count = retries + 1
                if c.defer_retry_count > max_retry:
                    c.status = "dropped_defer_exhausted"
                    return True, "retry_exhausted_dropped"
                c.status = "shadow"
                return True, "deferred"
        return False, "not_found"

    def get_active(self):
        for c in self.candidates:
            if c.status in ("unverified", "assigned_to_ugv", "visited"):
                return c
        return None

    def assign_to_ugv(self, candidate_id):
        for c in self.candidates:
            if c.candidate_id == candidate_id and c.status == "unverified":
                c.status = "assigned_to_ugv"
                return True
        return False

    def mark_visited(self, candidate_id):
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "visited"
                c.visited_by_ugv = True
                c.stop_observe_count += 1
                return True
        return False

    def record_stop_observation(self, candidate_id, step):
        """STOP_OBSERVE v2 语义（外审 P0-C02 修复，Phase-G2 Stage 0 Step G1）：
        STOP 是观察事件，不是候选生命周期转换——只累计 stop_observe_count 并更新
        last_observed_step，**不改 status**（visited 态两家 prompt 都不处理，曾把
        状态机推进死区）。CONFIRM/REJECT 才做生命周期裁决。mark_visited 保留仅为
        兼容旧调用方，runner 已不再使用。

        Returns:
            True 找到候选并记录；False 候选不存在。
        """
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.stop_observe_count += 1
                c.last_observed_step = step
                return True
        return False

    def confirm(self, candidate_id):
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "confirmed"
                c.confirm_attempts += 1
                return True
        return False

    def reject(self, candidate_id, step=None):
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "rejected"
                # REJECT does not consume a confirm attempt; confirm_attempts only increments on CONFIRM.
                ctr=(c.region_center_xy[0], c.region_center_xy[1])
                self.rejected_regions.append(ctr)
                if self.rejected_cooldown_steps is not None:
                    self._suppression_events.append({"center":ctr, "kind":"reject",
                                                     "step":int(step or 0), "reopens":0,
                                                     "candidate_id":candidate_id})
                return True
        return False

    def mark_unserviceable(self, candidate_id, reason=""):
        """监护逻辑专用：UGV 无法对该 candidate 做出裁决（巡逻死循环/导航停滞）时，
        由 runner 确定性清除 active 状态，使 UAV 解冻继续搜索。
        与 reject 一样加入抑制区，避免 UAV 立刻在同一区域重复标注。"""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.status = "unserviceable"
                c.reject_reason = reason or "unserviceable"
                ctr=(c.region_center_xy[0], c.region_center_xy[1])
                self.rejected_regions.append(ctr)
                if self.rejected_cooldown_steps is not None:
                    self._suppression_events.append({"center":ctr, "kind":"unserviceable",
                                                     "step":0, "reopens":0,
                                                     "candidate_id":candidate_id})
                return True
        return False

# ============================================================
# TopDown Map Renderer
# ============================================================
def _draw_dashed_rect(draw, x1, y1, x2, y2, color, dash_len=5, gap=3, width=1):
    """画虚线矩形"""
    for edge in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]:
        ex1, ey1, ex2, ey2 = edge
        dist = math.sqrt((ex2-ex1)**2 + (ey2-ey1)**2)
        if dist == 0:
            continue
        dx, dy = (ex2-ex1)/dist, (ey2-ey1)/dist
        pos = 0
        while pos < dist:
            sx = ex1 + dx * pos
            sy = ey1 + dy * pos
            ep = min(pos + dash_len, dist)
            ex = ex1 + dx * ep
            ey = ey1 + dy * ep
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos += dash_len + gap


def render_topdown(map_frame, uav_pos, ugv_pos, candidates, uav_traj=None, ugv_traj=None,
                   img_w=800, img_h=800, road_waypoints=None, road_graph_store=None,
                   background_image_bytes=None, show_road_network=True, region=None):
    """渲染 TopDown 地图（含道路网络背景，可选实际 RGB 俯视图作为底图）

    show_road_network: 是否绘制路网叠加层（道路、waypoint）。False 时只显示 RGB 背景 + UAV/UGV/候选标记。
    """
    # P0-fix（2026-07-21）：支持以实际 RGB 俯视图为背景，不是纯色
    # 2026-07-24 修正：capture_global_topdown yaw=90 时图像方向和 MapFrame 完全一致
    # （image right=world east, image down=world south = MapFrame px=y, py=x），不需要旋转。
    if background_image_bytes:
        try:
            bg = Image.open(io.BytesIO(background_image_bytes))
            bg = bg.resize((img_w, img_h), Image.Resampling.LANCZOS)
            img = bg.convert("RGB")
        except Exception:
            img = Image.new("RGB", (img_w, img_h), (25, 25, 30))
    else:
        img = Image.new("RGB", (img_w, img_h), (25, 25, 30))
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    # 0. Draw road network background (only if show_road_network=True)
    if show_road_network and road_graph_store is not None:
        # Draw real fine edges with direction
        import math
        for edge in road_graph_store.edges.values():
            src = road_graph_store.nodes.get(edge.src)
            dst = road_graph_store.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            x1, y1 = map_frame.world_to_pixel(src.x, src.y)
            x2, y2 = map_frame.world_to_pixel(dst.x, dst.y)
            color = (80, 80, 90) if edge.edge_type == "LANE_FOLLOW" else (120, 120, 60)
            if edge.edge_type.startswith("LANE_CHANGE"):
                color = (60, 120, 80)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
            # Arrow head
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d > 8:
                ux, uy = dx / d, dy / d
                px = x2 - ux * 6
                py = y2 - uy * 6
                perp = (-uy, ux)
                draw.polygon([(x2, y2), (px + perp[0]*2, py + perp[1]*2), (px - perp[0]*2, py - perp[1]*2)], fill=color)
    elif road_waypoints:
        # Fallback to old road_id grouping (deprecated)
        road_groups = {}
        for wp in road_waypoints:
            rid = wp.get("road_id", 0)
            if rid not in road_groups:
                road_groups[rid] = []
            road_groups[rid].append((wp["x"], wp["y"]))

        for rid, pts in road_groups.items():
            if len(pts) < 2:
                continue
            pixel_pts = [map_frame.world_to_pixel(p[0], p[1]) for p in pts]
            draw.line(pixel_pts, fill=(60, 60, 70, 200), width=4)
            draw.line(pixel_pts, fill=(80, 80, 90, 150), width=2)
    # 0.5 Draw task region boundary (gray dashed rectangle)
    if region is not None:
        rx1, ry1 = map_frame.world_to_pixel(region["x_min"], region["y_min"])
        rx2, ry2 = map_frame.world_to_pixel(region["x_max"], region["y_max"])
        _draw_dashed_rect(draw, rx1, ry1, rx2, ry2, (128, 128, 128, 180), dash_len=8, gap=4, width=3)

    # 0.6 Draw MapFrame coverage boundary (white dashed rectangle, smaller)
    # B19：内缩余量用模块常量（原硬编码 ±20）
    x1, y1 = map_frame.world_to_pixel(map_frame.x_min + TOPDOWN_RENDER_MARGIN_M,
                                      map_frame.y_min + TOPDOWN_RENDER_MARGIN_M)
    x2, y2 = map_frame.world_to_pixel(map_frame.x_max - TOPDOWN_RENDER_MARGIN_M,
                                      map_frame.y_max - TOPDOWN_RENDER_MARGIN_M)
    _draw_dashed_rect(draw, x1, y1, x2, y2, (255, 255, 255, 60), dash_len=5, gap=3, width=1)

    # Trajectories
    if uav_traj and len(uav_traj) > 1:
        pts = [map_frame.world_to_pixel(p[0], p[1]) for p in uav_traj]
        draw.line(pts, fill=(100, 150, 255, 80), width=1)
    if ugv_traj and len(ugv_traj) > 1:
        pts = [map_frame.world_to_pixel(p[0], p[1]) for p in ugv_traj]
        draw.line(pts, fill=(100, 255, 150, 80), width=1)

    # Candidates
    for c in candidates:
        if c.status in ("unverified", "assigned_to_ugv", "visited"):
            color = (255, 165, 0, 100)
            border = (255, 165, 0, 255)
            border_width = 3 if c.status == "assigned_to_ugv" else 1
        elif c.status == "rejected":
            color = (128, 128, 128, 60)
            border = (128, 128, 128, 255)
            border_width = 1
        elif c.status == "confirmed":
            color = (255, 0, 0, 100)
            border = (255, 0, 0, 255)
            border_width = 2
        else:
            continue

        if len(c.region_polygon) >= 3:
            pts = [map_frame.world_to_pixel(p[0], p[1]) for p in c.region_polygon]
            draw.polygon(pts, fill=color, outline=border, width=border_width)

        cx, cy = map_frame.world_to_pixel(c.region_center_xy[0], c.region_center_xy[1])
        draw.text((cx - 10, cy - 5), c.candidate_id, fill=border[:3], font=font)
        if c.status == "rejected":
            draw.line([cx-8, cy-8, cx+8, cy+8], fill=(255, 0, 0), width=2)
            draw.line([cx-8, cy+8, cx+8, cy-8], fill=(255, 0, 0), width=2)
        elif c.status == "confirmed":
            # Draw checkmark
            draw.line([cx-6, cy, cx-2, cy+4], fill=(0, 255, 0), width=2)
            draw.line([cx-2, cy+4, cx+6, cy-4], fill=(0, 255, 0), width=2)

    # UAV position (blue)
    ux, uy = map_frame.world_to_pixel(uav_pos[0], uav_pos[1])
    draw.ellipse([ux-5, uy-5, ux+5, uy+5], fill=(0, 100, 255))
    draw.text((ux + 6, uy - 5), "UAV", fill=(0, 150, 255), font=font)

    # UGV position (green)
    gx, gy = map_frame.world_to_pixel(ugv_pos[0], ugv_pos[1])
    draw.ellipse([gx-5, gy-5, gx+5, gy+5], fill=(0, 255, 100))
    draw.text((gx + 6, gy - 5), "UGV", fill=(0, 255, 150), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ============================================================
# Unified Camera Projection (P0-3)
# ============================================================
def _camera_basis(yaw_deg, pitch_deg):
    """Return (forward, right, up) unit vectors for a CARLA camera.

    Uses CARLA's own Rotation math so we stay 100% consistent with the
    simulator's camera placement (left-handed, Z-up; pitch=-90 looks down).

    Camera frame: x=forward, y=right, z=up. Image pixel x grows right,
    pixel y grows DOWN, so image_y uses -z_cam.
    """
    import carla
    rot = carla.Rotation(yaw=float(yaw_deg), pitch=float(pitch_deg))
    f = rot.get_forward_vector()
    r = rot.get_right_vector()
    u = rot.get_up_vector()
    return (f.x, f.y, f.z), (r.x, r.y, r.z), (u.x, u.y, u.z)


def project_point_to_image(world_pt, cam_loc, yaw_deg, pitch_deg, fov_deg, img_w, img_h):
    """Project a single world point to image pixel coordinates.

    Returns (px, py, depth) where px/py are floats (may be outside [0,w]/[0,h])
    and depth is distance along the camera forward axis (<=0 means behind).
    """
    f, r, u = _camera_basis(yaw_deg, pitch_deg)
    dx = world_pt[0] - cam_loc[0]
    dy = world_pt[1] - cam_loc[1]
    dz = world_pt[2] - cam_loc[2]
    x_cam = dx * f[0] + dy * f[1] + dz * f[2]   # forward depth
    y_cam = dx * r[0] + dy * r[1] + dz * r[2]   # right
    z_cam = dx * u[0] + dy * u[1] + dz * u[2]   # up
    if x_cam <= 1e-4:
        return None, None, x_cam
    fx = (img_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    px = fx * (y_cam / x_cam) + img_w / 2.0
    py = -fx * (z_cam / x_cam) + img_h / 2.0
    return px, py, x_cam


def project_actor_bbox_to_image(target_actor, cam_loc, yaw_deg, pitch_deg,
                                fov_deg, img_w, img_h):
    """Project an actor's 3D bounding box to a 2D image bbox.

    Returns dict with:
      ok          - True if any corner in front of camera
      bbox_2d     - [min_px, min_py, max_px, max_py] (unclamped floats)
      corners_px  - list of (px, py, depth) for the 8 corners
    """
    import carla
    bbox = target_actor.bounding_box
    transform = target_actor.get_transform()
    ext = bbox.extent
    bb_loc = bbox.location  # bbox center in actor frame (P0-3)
    corners_world = []
    for dx, dy, dz in [(ext.x, ext.y, ext.z), (-ext.x, ext.y, ext.z),
                       (ext.x, -ext.y, ext.z), (-ext.x, -ext.y, ext.z),
                       (ext.x, ext.y, -ext.z), (-ext.x, ext.y, -ext.z),
                       (ext.x, -ext.y, -ext.z), (-ext.x, -ext.y, -ext.z)]:
        wl = transform.transform(carla.Location(
            x=bb_loc.x + dx, y=bb_loc.y + dy, z=bb_loc.z + dz))
        corners_world.append((wl.x, wl.y, wl.z))

    corners_px = []
    pxs, pys = [], []
    for cw in corners_world:
        px, py, depth = project_point_to_image(cw, cam_loc, yaw_deg, pitch_deg,
                                               fov_deg, img_w, img_h)
        corners_px.append((px, py, depth))
        if px is not None:
            pxs.append(px)
            pys.append(py)
    if not pxs:
        return {"ok": False, "bbox_2d": None, "corners_px": corners_px}
    return {
        "ok": True,
        "bbox_2d": [min(pxs), min(pys), max(pxs), max(pys)],
        "corners_px": corners_px,
    }


def check_visibility_unified(target_actor, cam_loc, yaw_deg, pitch_deg,
                             fov_deg, img_w, img_h, carla_world,
                             min_long_side_px=12, min_area_ratio=0.0005,
                             min_margin_px=0, los_fail_closed=False):
    """Unified visibility check for ANY camera pitch/yaw (P0-3).

    Quality gates (all must pass to return visible=True):
      1. bbox intersects the image frame
      2. bbox long side >= min_long_side_px
      3. bbox area ratio >= min_area_ratio
      4. bbox center keeps >= min_margin_px from every image edge
      5. LOS raycast clear (fail-open only when raycast API unavailable)

    B3（2026-09-17 审计）：los_fail_closed=True 时 raycast API 异常视为不通视
    （fail-closed，debug 记 los_raycast_error_fail_closed）；默认 False=旧行为
    （fail-open，避免 API 故障阻断所有 episode）。由 runner 按
    perception.los_fail_closed config 传入。
    Returns (visible: bool, info: dict).
    """
    import carla
    try:
        proj = project_actor_bbox_to_image(target_actor, cam_loc, yaw_deg,
                                           pitch_deg, fov_deg, img_w, img_h)
        if not proj["ok"]:
            return False, {"reason": "all_behind_camera"}
        min_px, min_py, max_px, max_py = proj["bbox_2d"]

        # 1. frame intersection
        if max_px < 0 or min_px > img_w or max_py < 0 or min_py > img_h:
            return False, {"reason": "out_of_frame",
                           "bbox_2d": proj["bbox_2d"]}
        # clamp for size metrics
        c_min_px, c_min_py = max(0.0, min_px), max(0.0, min_py)
        c_max_px, c_max_py = min(float(img_w), max_px), min(float(img_h), max_py)

        # 2+3. size gates
        long_side = max(c_max_px - c_min_px, c_max_py - c_min_py)
        area_ratio = ((c_max_px - c_min_px) * (c_max_py - c_min_py)) / float(img_w * img_h)
        if long_side < min_long_side_px or area_ratio < min_area_ratio:
            return False, {"reason": "too_small",
                           "long_side_px": long_side,
                           "area_ratio": area_ratio,
                           "bbox_2d": proj["bbox_2d"]}

        # 4. edge margin on bbox CENTER (unclamped center)
        cx = (min_px + max_px) / 2.0
        cy = (min_py + max_py) / 2.0
        if min_margin_px > 0:
            if (cx < min_margin_px or cx > img_w - min_margin_px or
                    cy < min_margin_px or cy > img_h - min_margin_px):
                return False, {"reason": "center_near_edge",
                               "center_px": [cx, cy],
                               "bbox_2d": proj["bbox_2d"]}

        # 5. LOS raycast (same sampling strategy as legacy checker)
        los_clear = False
        los_debug = []
        try:
            eye = carla.Location(x=cam_loc[0], y=cam_loc[1], z=cam_loc[2])
            tf = target_actor.get_transform()
            # P0-3: sample the vehicle's actual bbox center height, not the
            # pivot (vehicle pivots sit near z=0/road level while the bbox
            # floats above it).
            bbox = target_actor.bounding_box
            bb_loc = bbox.location  # bbox center in actor frame
            ext = bbox.extent
            center_w = tf.transform(bb_loc)
            sample_pts = [center_w]
            for dx, dy, dz in [(ext.x, ext.y, ext.z), (-ext.x, ext.y, ext.z),
                               (ext.x, -ext.y, ext.z), (-ext.x, -ext.y, ext.z)]:
                sample_pts.append(tf.transform(carla.Location(
                    x=bb_loc.x + dx, y=bb_loc.y + dy, z=bb_loc.z + dz)))
            for j, tgt_pt in enumerate(sample_pts):
                hits = carla_world.cast_ray(eye, tgt_pt)
                if hits is None:
                    # Fail-closed must cover an unavailable raycast result as well
                    # as an exception. R22 handled exceptions only.
                    los_debug.append({"idx": j, "hits": None,
                                      "reason": "raycast_unavailable"})
                    los_clear = not los_fail_closed
                    break
                first_non_target = None
                dist_to_tgt = math.sqrt((tgt_pt.x - eye.x) ** 2 +
                                        (tgt_pt.y - eye.y) ** 2 +
                                        (tgt_pt.z - eye.z) ** 2)
                for h in hits:
                    hl = h.location
                    dist_to_hit = math.sqrt((hl.x - eye.x) ** 2 +
                                            (hl.y - eye.y) ** 2 +
                                            (hl.z - eye.z) ** 2)
                    # Skip hits beyond or at target point
                    if dist_to_hit >= dist_to_tgt - 0.5:
                        continue
                    # P0-fix (2026-07-21): skip hits belonging to target actor itself
                    hit_actor = getattr(h, "actor", None)
                    if hit_actor is None:
                        hit_actor = getattr(h, "get_actor", lambda: None)()
                    hit_actor_id = getattr(hit_actor, "id", None) if hit_actor else None
                    if hit_actor_id is not None and hit_actor_id == target_actor.id:
                        continue
                    first_non_target = h
                    break
                if first_non_target is None:
                    los_clear = True
                    los_debug.append({"idx": j, "hits": len(hits)})
                    break
                else:
                    los_debug.append({"idx": j,
                                      "hit_label": str(getattr(first_non_target, "label", None))})
        except Exception as e:
            logger.warning(f"LOS raycast failed (unified): {e}")
            if los_fail_closed:
                # B3：fail-closed——raycast 异常视为不通视，并在 debug 里记录原因
                los_clear = False
                los_debug.append({"reason": "los_raycast_error_fail_closed", "error": str(e)})
            else:
                los_clear = True  # fail-open on API error（旧行为）

        # P0-2 (2026-09-04 external review, user-approved): visible 现在受 LOS 控制
        # （visible = 投影 gate 全过 AND los_clear）。投影可见单独记录在
        # projected_visible 字段（reference 生成器的"LOS 软门"逻辑改用该字段）。
        projected_visible = True  # 走到这里说明 frame/size/margin 投影 gate 已全过
        visible = projected_visible and los_clear
        return visible, {
            "reason": "visible" if visible else "los_blocked",
            "projected_visible": projected_visible,
            "bbox_2d": proj["bbox_2d"],
            "long_side_px": long_side,
            "area_ratio": area_ratio,
            "los_clear": los_clear,
            "los_debug": los_debug,
        }
    except Exception as e:
        logger.error(f"check_visibility_unified error: {e}")
        return False, {"reason": "error", "error": str(e)}


# ============================================================
# Visibility Checker
# ============================================================
def check_visibility(target_actor, camera_loc, camera_yaw, camera_fov, img_w, img_h, carla_world,
                     observer_actor=None, los_fail_closed=False):
    """检查目标是否在相机视野中可见（bbox 投影 + raycast LOS）
    
    Args:
        observer_actor: UGV/observer 自身的 actor，用于在 LOS raycast 中过滤掉自身几何体。
                       如果不传，则只过滤 target_actor（可能导致 observer 自身被误判为遮挡）。
        los_fail_closed: B3（2026-09-17 审计）。True 时 raycast API 异常视为不通视
                       （fail-closed）；默认 False=旧行为（fail-open）。
    """
    import carla
    try:
        bbox = target_actor.bounding_box
        transform = target_actor.get_transform()
        target_loc = transform.location

        # 3D bbox corners
        corners_3d = []
        ext = bbox.extent
        for dx, dy, dz in [(ext.x, ext.y, ext.z), (-ext.x, ext.y, ext.z),
                           (ext.x, -ext.y, ext.z), (-ext.x, -ext.y, ext.z),
                           (ext.x, ext.y, -ext.z), (-ext.x, ext.y, -ext.z),
                           (ext.x, -ext.y, -ext.z), (-ext.x, -ext.y, -ext.z)]:
            world_loc = transform.transform(carla.Location(x=dx, y=dy, z=dz))
            corners_3d.append(world_loc)

        # Project to 2D
        cam_loc = carla.Location(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2])
        # Compute relative positions
        rel_corners = []
        for c in corners_3d:
            dx = c.x - cam_loc.x
            dy = c.y - cam_loc.y
            dz = c.z - cam_loc.z
            # Rotate by -yaw to camera frame
            yaw_rad = math.radians(camera_yaw)
            cx = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            cy = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            cz = dz
            rel_corners.append((cx, cy, cz))

        # For each corner, check if in front (cx > 0 for forward-facing camera)
        # For down-looking camera (pitch=-90), we need different logic
        # Simplified: check if target is within FOV cone
        target_dist = math.sqrt((corners_3d[0].x - cam_loc.x)**2 + (corners_3d[0].y - cam_loc.y)**2 + (corners_3d[0].z - cam_loc.z)**2)
        if target_dist == 0:
            return False, {"reason": "zero_distance"}

        # Check angle to target center
        dx = target_loc.x - cam_loc.x
        dy = target_loc.y - cam_loc.y
        dz = target_loc.z - cam_loc.z
        angle_to_target = math.degrees(math.atan2(math.sqrt(dx**2 + dy**2), abs(dz)))

        # For UGV forward camera (pitch=0, yaw=camera_yaw):
        # Check horizontal angle
        cam_yaw_rad = math.radians(camera_yaw)
        rel_x = dx * math.cos(cam_yaw_rad) + dy * math.sin(cam_yaw_rad)
        rel_y = -dx * math.sin(cam_yaw_rad) + dy * math.cos(cam_yaw_rad)
        horizontal_angle = math.degrees(math.atan2(abs(rel_y), rel_x)) if rel_x > 0 else 90

        if horizontal_angle > camera_fov / 2:
            return False, {"reason": "out_of_fov"}

        if rel_x < 0:
            return False, {"reason": "behind_camera"}

        # Compute 2D bbox
        fx = (img_w / 2) / math.tan(math.radians(camera_fov) / 2)
        px_list = []
        for cx, cy, cz in rel_corners:
            if cx > 0.1:
                px = fx * cy / cx + img_w / 2
                py = fx * cz / cx + img_h / 2  # simplified
                px_list.append((px, py))

        if not px_list:
            return False, {"reason": "all_behind"}

        min_px = min(p[0] for p in px_list)
        max_px = max(p[0] for p in px_list)
        min_py = min(p[1] for p in px_list)
        max_py = max(p[1] for p in px_list)

        # Check bbox in image
        if max_px < 0 or min_px > img_w or max_py < 0 or min_py > img_h:
            return False, {"reason": "out_of_frame"}

        # Check bbox size
        long_side = max(max_px - min_px, max_py - min_py)
        area_ratio = (max_px - min_px) * (max_py - min_py) / (img_w * img_h)
        if long_side < 12 or area_ratio < 0.0005:
            return False, {"reason": "too_small", "long_side": long_side, "area_ratio": area_ratio}

        # LOS raycast: sample multiple points on target bbox and check occlusion
        # P0-fix (2026-07-22 用户要求): 方案 C —— 中心点优先，否则至少 2 个角通视。
        # 避免"目标 90% 可见但一个角被灯杆挡住"就误判为遮挡，
        # 同时避免"目标只有 10% 可见（一个角）"就误判为通视。
        los_clear = False
        los_debug = []
        try:
            # Sample observer eye-level and target points (center + bbox corners)
            eye = carla.Location(x=cam_loc.x, y=cam_loc.y, z=cam_loc.z)
            sample_points = [target_loc] + corners_3d[:4]

            # 方案B (2026-07-22)：识别"目标车自身低矮部位"与"路缘/地面凸起"造成的假遮挡。
            # 目标车(尤其 truck)的轮子/底盘是 mesh 一部分, raycast 命中时 h.actor 可能为 None
            # 或 id 对不上, 导致"跳过目标自身"失效, 把目标自身轮子误判为遮挡 -> 假阴性。
            # 判定: 命中点 z 接近"目标车底部 z"(轮子/底盘高度) -> 视为目标自身部位, 跳过。
            _tgt_bottom_z = None
            try:
                _bb = target_actor.bounding_box
                _tf = target_actor.get_transform()
                _c = _tf.transform(_bb.location)
                _tgt_bottom_z = _c.z - _bb.extent.z   # 目标车底面 z
            except Exception:
                _tgt_bottom_z = None

            def _is_point_clear(tgt_pt):
                """检查单个采样点是否通视（无遮挡）。返回 (is_clear, debug_info)"""
                # B3（2026-09-17 审计）：三个启发式跳过分支（贴眼<2m / label=NONE 近地 /
                # 底盘<0.8m）任一命中时在 debug 输出标 los_waived=True（增量字段，
                # 供离线敏感性分析统计"若不豁免该命中则遮挡"的影响面）
                waived = False
                hits = carla_world.cast_ray(eye, tgt_pt)
                if hits is None:
                    return (not los_fail_closed), {
                        "hits": None,
                        "reason": ("raycast_unavailable_fail_closed"
                                   if los_fail_closed else "raycast_unavailable"),
                        "los_waived": waived}
                first_non_target = None
                dist_to_tgt = math.sqrt((tgt_pt.x - eye.x)**2 + (tgt_pt.y - eye.y)**2 + (tgt_pt.z - eye.z)**2)
                for h in hits:
                    hl = h.location
                    dist_to_hit = math.sqrt((hl.x - eye.x)**2 + (hl.y - eye.y)**2 + (hl.z - eye.z)**2)
                    # Skip hits that are beyond or at the target point (within 1.0m tolerance)
                    if dist_to_hit >= dist_to_tgt - 1.0:
                        continue
                    # Skip hits that belong to the target actor itself (not occlusion)
                    hit_actor = getattr(h, "actor", None)
                    if hit_actor is None:
                        hit_actor = getattr(h, "get_actor", lambda: None)()
                    hit_actor_id = getattr(hit_actor, "id", None) if hit_actor else None
                    if hit_actor_id is not None and hit_actor_id == target_actor.id:
                        continue
                    # Skip hits belonging to observer (UGV) itself
                    if observer_actor is not None and hit_actor_id is not None and hit_actor_id == observer_actor.id:
                        continue
                    label = getattr(h, "label", None)
                    label_name = str(label) if label is not None else ""
                    # Skip hits with label "Vehicles" that are the target itself (redundant but safe)
                    if label_name == "Vehicles" and hit_actor_id == target_actor.id:
                        continue
                    # 方案B-0: 命中点距离 eye 极近(<2m) -> 眼位贴脸命中(observer 自身 mesh/贴脸物体)。
                    # cast_ray 起点在 observer mesh 内部/表面时, 会在出发点 几微米~2m 处命中
                    # 无标签无 actor 的自身几何, 这不是真遮挡, 必须跳过(否则所有采样点全被它挡死)。
                    if dist_to_hit < 2.0 and hit_actor_id is None:
                        waived = True
                        continue
                    # 方案B-1: label=NONE 且命中点接近地面 -> 路缘/地面凸起/车轮拱, 非真遮挡
                    if label_name in ("", "NONE", "None"):
                        # 命中点接近地面(eye 地面或目标底部附近) -> 跳过
                        if _tgt_bottom_z is not None and abs(hl.z - _tgt_bottom_z) < 1.0:
                            waived = True
                            continue
                        if hl.z < (eye.z - 1.2):   # 明显低于眼位 -> 地面/路缘
                            waived = True
                            continue
                    # 方案B-2: 命中点接近目标车底部(轮子/底盘) -> 目标自身部位, 跳过
                    # (轮子/底盘是 target mesh 一部分但 actor id 对不上)
                    if _tgt_bottom_z is not None and abs(hl.z - _tgt_bottom_z) < 0.8 \
                            and label_name in ("Vehicles", "Truck", "Car", ""):
                        waived = True
                        continue
                    first_non_target = h
                    break
                if first_non_target is None:
                    return True, {"hits": len(hits), "reason": "clear", "los_waived": waived}
                else:
                    label = getattr(first_non_target, "label", None)
                    return False, {"hits": len(hits), "hit_label": str(label), "hit_dist": dist_to_hit,
                                   "reason": "occluded", "los_waived": waived}

            # 方案 C：中心点优先
            center_clear, center_debug = _is_point_clear(sample_points[0])
            los_debug.append({"idx": 0, "point": (sample_points[0].x, sample_points[0].y, sample_points[0].z),
                              "type": "center", "clear": center_clear, **center_debug})
            if center_clear:
                los_clear = True  # 中心通视，直接 True（目标主体可见）
            else:
                # 中心被挡，检查 4 个角：至少 2 个角通视才算通视
                corner_clears = []
                for j, corner_pt in enumerate(sample_points[1:], start=1):
                    c_clear, c_debug = _is_point_clear(corner_pt)
                    corner_clears.append(c_clear)
                    los_debug.append({"idx": j, "point": (corner_pt.x, corner_pt.y, corner_pt.z),
                                      "type": "corner", "clear": c_clear, **c_debug})
                clear_corner_count = sum(corner_clears)
                los_clear = (clear_corner_count >= 2)  # 至少 2 个角通视
        except Exception as e:
            logger.warning(f"LOS raycast failed: {e}")
            if los_fail_closed:
                # B3：fail-closed——raycast 异常视为不通视，并在 debug 里记录原因
                los_clear = False
                los_debug.append({"reason": "los_raycast_error_fail_closed", "error": str(e)})
            else:
                los_clear = True  # fail-open on API error to avoid blocking all episodes（旧行为）

        # P0-2 (2026-09-04): 与 check_visibility_unified 同步——visible 受 LOS 控制，
        # 投影可见单独记录在 projected_visible。
        projected_visible = True  # 投影 gate 已全过
        visible = projected_visible and los_clear
        return visible, {
            "reason": "visible" if visible else "los_blocked",
            "projected_visible": projected_visible,
            "bbox_2d": [min_px, min_py, max_px, max_py],
            "long_side_px": long_side,
            "area_ratio": area_ratio,
            "los_clear": los_clear,
            "distance": target_dist,
            "los_debug": los_debug,
        }
    except Exception as e:
        logger.error(f"Visibility check error: {e}")
        return False, {"reason": "error", "error": str(e)}

def check_visibility_6view(target_actor, ugv_loc, ugv_yaw, yaws, fov, img_w, img_h, carla_world,
                           observer_actor=None, eye_z=None, los_fail_closed=False):
    """检查 6 视图中目标是否可见，返回 (any_visible, visible_views, per_view_results, los_clear)

    Args:
        observer_actor: UGV 自身的 actor，用于在 LOS raycast 中过滤掉自身几何体。
        eye_z: 可见性判定的相机眼位高度（世界坐标 z）。
            2026-07-22 方案A：必须与实际拍照相机一致（UGV 车顶+0.3m），
            不再写死 1.8——低眼位对高目标(truck)会被路缘/车轮/底盘大量误挡。
            不传时默认用 observer_actor 车顶+0.3m；再退化到 1.8。
        los_fail_closed: B3（2026-09-17 审计）。透传给单视图 check_visibility，
            True 时 raycast API 异常视为不通视；默认 False=旧行为（fail-open）。
    """
    # R33: actual RGB cameras use the transformed bounding-box center in XY
    # as well as roof+0.3m in Z. Actor origins can be off-center and their offset
    # rotates with yaw; using origin XY here would certify a different camera.
    eye_xy = ugv_loc
    if observer_actor is not None:
        try:
            _center = observer_actor.get_transform().transform(observer_actor.bounding_box.location)
            eye_xy = (_center.x, _center.y)
        except Exception:
            pass
    # 方案A：统一眼位高度 = observer 车顶 + 0.3m（与实际拍照相机同源）
    if eye_z is None:
        eye_z = 1.8
        if observer_actor is not None:
            try:
                _tf = observer_actor.get_transform()
                _bb = observer_actor.bounding_box
                _cw = _tf.transform(_bb.location)
                eye_z = _cw.z + _bb.extent.z + 0.3
            except Exception:
                eye_z = 1.8
    results = []
    any_visible = False
    visible_views = []
    los_clear = False
    for yaw_offset in yaws:
        cam_yaw = ugv_yaw + yaw_offset
        cam_loc = (eye_xy[0], eye_xy[1], eye_z)
        visible, debug = check_visibility(
            target_actor, cam_loc, cam_yaw, fov, img_w, img_h, carla_world,
            observer_actor=observer_actor, los_fail_closed=los_fail_closed
        )
        results.append({"yaw": yaw_offset, "visible": visible, "debug": debug})
        if visible:
            any_visible = True
            visible_views.append(yaw_offset)
            if debug.get("los_clear", False):
                los_clear = True
    return any_visible, visible_views, results, los_clear


# ============================================================
# 演示素材增强渲染（2026-09-13 R12 固化；用户认可的设计：
# VLM 输入保持原始渲染，仅"存图"走增强版——防目标真值泄漏）
# ============================================================

def _star_polygon(cx, cy, r_out, r_in):
    """五角星顶点（外接/内切半径）。"""
    import math
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def render_topdown_annotated(map_frame, uav_pos, ugv_pos, candidates, uav_traj=None, ugv_traj=None,
                             img_w=800, img_h=800, road_waypoints=None, road_graph_store=None,
                             background_image_bytes=None, show_road_network=True, region=None,
                             target_pos=None):
    """演示素材增强版 topdown（R12 2026-09-13 用户要求）。

    与 render_topdown 的差异：UAV/UGV 图标放大加白描边、字体加大加粗、
    双机轨迹加粗（辉光+芯线）、候选标注放大、真实目标位置画红色五角星 +
    红色 "Target" 文字。**仅用于人看的演示存图，绝不能作为 VLM 输入**
    （目标真值位置对模型是泄漏）。
    """
    if background_image_bytes:
        try:
            bg = Image.open(io.BytesIO(background_image_bytes))
            bg = bg.resize((img_w, img_h), Image.Resampling.LANCZOS)
            img = bg.convert("RGB")
        except Exception:
            img = Image.new("RGB", (img_w, img_h), (25, 25, 30))
    else:
        img = Image.new("RGB", (img_w, img_h), (25, 25, 30))
    draw = ImageDraw.Draw(img, "RGBA")
    font = None
    for fp, sz in (("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22),
                   ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22),
                   ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)):
        try:
            font = ImageFont.truetype(fp, sz)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    # 任务区域边界
    if region is not None:
        rx1, ry1 = map_frame.world_to_pixel(region["x_min"], region["y_min"])
        rx2, ry2 = map_frame.world_to_pixel(region["x_max"], region["y_max"])
        _draw_dashed_rect(draw, rx1, ry1, rx2, ry2, (128, 128, 128, 200), dash_len=10, gap=5, width=4)

    # 轨迹：辉光 + 芯线双描（明显加粗）
    if uav_traj and len(uav_traj) > 1:
        pts = [map_frame.world_to_pixel(p[0], p[1]) for p in uav_traj]
        draw.line(pts, fill=(100, 150, 255, 90), width=11)
        draw.line(pts, fill=(80, 130, 255, 235), width=5)
    if ugv_traj and len(ugv_traj) > 1:
        pts = [map_frame.world_to_pixel(p[0], p[1]) for p in ugv_traj]
        draw.line(pts, fill=(100, 255, 150, 90), width=11)
        draw.line(pts, fill=(60, 235, 120, 235), width=5)

    # 候选区域（放大版）
    for c in candidates:
        if c.status in ("unverified", "assigned_to_ugv", "visited"):
            color = (255, 165, 0, 100)
            border = (255, 165, 0, 255)
            border_width = 6 if c.status == "assigned_to_ugv" else 3
        elif c.status == "rejected":
            color = (128, 128, 128, 60)
            border = (128, 128, 128, 255)
            border_width = 2
        elif c.status == "confirmed":
            color = (255, 0, 0, 100)
            border = (255, 0, 0, 255)
            border_width = 5
        else:
            continue
        if len(c.region_polygon) >= 3:
            pts = [map_frame.world_to_pixel(p[0], p[1]) for p in c.region_polygon]
            draw.polygon(pts, fill=color, outline=border, width=border_width)
        cx, cy = map_frame.world_to_pixel(c.region_center_xy[0], c.region_center_xy[1])
        draw.text((cx - 14, cy - 11), c.candidate_id, fill=border[:3], font=font)
        if c.status == "rejected":
            draw.line([cx - 11, cy - 11, cx + 11, cy + 11], fill=(255, 0, 0), width=4)
            draw.line([cx - 11, cy + 11, cx + 11, cy - 11], fill=(255, 0, 0), width=4)
        elif c.status == "confirmed":
            draw.line([cx - 9, cy, cx - 3, cy + 6], fill=(0, 255, 0), width=5)
            draw.line([cx - 3, cy + 6, cx + 9, cy - 6], fill=(0, 255, 0), width=5)

    # 真实目标：红色五角星 + 红色 "Target" 文字
    if target_pos is not None:
        tx, ty = map_frame.world_to_pixel(target_pos[0], target_pos[1])
        star = _star_polygon(tx, ty, r_out=20, r_in=8)
        draw.polygon(star, fill=(230, 0, 0, 255), outline=(255, 255, 255, 255))
        draw.text((tx + 18, ty - 14), "Target", fill=(255, 40, 40), font=font)

    # UAV（蓝色，放大加白描边）
    ux, uy = map_frame.world_to_pixel(uav_pos[0], uav_pos[1])
    draw.ellipse([ux - 12, uy - 12, ux + 12, uy + 12], fill=(0, 100, 255), outline=(255, 255, 255), width=4)
    draw.ellipse([ux - 3, uy - 3, ux + 3, uy + 3], fill=(255, 255, 255))
    draw.text((ux + 16, uy - 13), "UAV", fill=(0, 150, 255), font=font)

    # UGV（绿色，放大加白描边）
    gx, gy = map_frame.world_to_pixel(ugv_pos[0], ugv_pos[1])
    draw.ellipse([gx - 12, gy - 12, gx + 12, gy + 12], fill=(0, 255, 100), outline=(255, 255, 255), width=4)
    draw.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(255, 255, 255))
    draw.text((gx + 16, gy - 13), "UGV", fill=(0, 255, 150), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
