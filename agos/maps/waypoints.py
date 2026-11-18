"""waypoints.py — UAV 网格 + UGV 路网航点 + Approach 选择器 + 地图坐标转换"""
import math, logging
from typing import List, Tuple, Optional

logger = logging.getLogger("LiteV1.Waypoints")

class MapFrame:
    """世界坐标 ↔ 地图像素坐标转换

    2026-07-24 修正：与 capture_global_topdown 使用完全相同的覆盖范围。
    capture_global_topdown 的 span = max(span_x, span_y) * 1.1（10% margin），
    所以 MapFrame 也应该使用相同的 span 作为覆盖范围，而不是 x_range = x_max - x_min。
    """
    def __init__(self, region, img_w=800, img_h=800, margin=20):
        # B19（2026-09-17 审计）：margin 为历史死参数（本函数从未使用）。grep 调用方
        # 发现 benchmark/testset 旧生成脚本仍按位/关键字传入 margin=0/30（外部依赖，
        # benchmark/ 不在本仓改动权限内），故按审计预案的条件分支保留签名不删除；
        # 禁止新增依赖。渲染内缩余量已统一为 perception.TOPDOWN_RENDER_MARGIN_M。
        self.img_w = img_w
        self.img_h = img_h
        # 与 capture_global_topdown 使用完全相同的覆盖范围（span * 1.1）
        span_x = region["x_max"] - region["x_min"]
        span_y = region["y_max"] - region["y_min"]
        span = max(span_x, span_y) * 1.1
        cx = (region["x_min"] + region["x_max"]) / 2.0
        cy = (region["y_min"] + region["y_max"]) / 2.0
        self.x_min = cx - span / 2.0
        self.x_max = cx + span / 2.0
        self.y_min = cy - span / 2.0
        self.y_max = cy + span / 2.0
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

    def world_to_pixel(self, x, y):
        """CARLA (x=front, y=right) → image (px=right=y, py=up=x)

        2026-07-24 修正：py 映射反转，使图像向上=世界北（x 越大越靠上）。
        之前 py = (x - x_min) / x_range * img_h 导致图像向下=世界北（x 越大越靠下），
        与"上北下南"的直觉相反。修正为 py = (x_max - x) / x_range * img_h。
        """
        px = int((y - self.y_min) / self.y_range * self.img_w)
        py = int((self.x_max - x) / self.x_range * self.img_h)
        return px, py

    def pixel_to_world(self, px, py):
        y = self.y_min + px / self.img_w * self.y_range
        x = self.x_max - py / self.img_h * self.x_range
        return x, y

class UAVGridCatalog:
    def __init__(self, waypoints, current_xy, spacing=40, region=None):
        self.waypoints = waypoints
        self.current = current_xy
        self.spacing = spacing
        from agos.benchmark_region import region_bounds
        self.region = region_bounds(region) if region is not None else None

    def _eligible(self, wp):
        from agos.benchmark_region import inside_task_region
        return bool(wp.get('active', True)) and (self.region is None or
            inside_task_region((wp['x'], wp['y']), self.region))

    def is_legal_waypoint(self, waypoint_id):
        return waypoint_id in {w['waypoint_id'] for w in self.get_available()}

    def get_available(self, max_count=8):
        """返回当前UAV周围一圈的8个网格航点（3×3减去中心）"""
        # 找到最近的网格航点作为基准
        nearest = None
        nearest_d = 999
        for wp in self.waypoints:
            if not self._eligible(wp):
                continue
            d = math.sqrt((wp["x"] - self.current[0])**2 + (wp["y"] - self.current[1])**2)
            if d < nearest_d:
                nearest_d = d
                nearest = wp

        if not nearest:
            return []

        # 找周围8个邻居（间距±spacing）
        cx, cy = nearest["x"], nearest["y"]
        available = []
        for wp in self.waypoints:
            if not self._eligible(wp):
                continue
            if wp["id"] == nearest["id"]:
                continue
            dx = wp["x"] - cx
            dy = wp["y"] - cy
            # 只选周围一圈（±1个间距）
            if abs(dx) <= self.spacing * 1.5 and abs(dy) <= self.spacing * 1.5:
                d = math.sqrt((wp["x"] - self.current[0])**2 + (wp["y"] - self.current[1])**2)
                # 2026-07-24: 计算方向（CARLA 坐标：X=北，Y=东），加浮点数容差避免精度问题
                dx = wp["x"] - self.current[0]
                dy = wp["y"] - self.current[1]
                eps = 1.0  # 1m 容差，避免浮点数精度导致方向误判
                dir_x = "N" if dx > eps else ("S" if dx < -eps else "")
                dir_y = "E" if dy > eps else ("W" if dy < -eps else "")
                direction = dir_x + dir_y if dir_x + dir_y else "C"
                available.append({
                    "waypoint_id": wp["id"],
                    "world_xy": (wp["x"], wp["y"]),
                    "distance_m": round(d, 1),
                    "visited": wp.get("visited", False),
                    "direction": direction,
                    "description": f"{direction} ({wp['x']:.0f}, {wp['y']:.0f})",
                })

        available.sort(key=lambda w: (w["visited"], w["distance_m"]))
        return available[:max_count]

    def mark_visited(self, waypoint_id):
        for wp in self.waypoints:
            if wp["id"] == waypoint_id:
                wp["visited"] = True

# ============================================================
# 2026-08-02 清理（bug review / 死代码删除）：
# 以下两类实现已被 roadnet 适配层（UGVRoadCatalogAdapter / ApproachAdapter，
# 基于冻结两层路网图）全面取代，且旧实现带有已知缺陷，保留只会诱导二次
# 开发误用，故整体删除：
#   - UGVRoadCatalog：左/右转 maneuver 标签疑似镜像（angle>0 标"左转"，
#     但在 +X北/+Y东 系下朝北时 +Y（angle=+90°）在右手边）；visited 字段
#     硬编码 False，防振荡提示失效。
#   - select_approach_waypoint：第三级 fallback 无距离上界，可返回 200m 外
#     航点，违反 ≤max_confirm_distance 的确认契约。
# 正式路径请使用：
#   from agos.roadnet import UGVRoadCatalogAdapter, ApproachAdapter
# ============================================================
