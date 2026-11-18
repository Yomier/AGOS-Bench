"""roadnet/locator.py - 世界坐标/UGV 位姿 -> 图节点定位 (v2)

关键改进（按外部 Agent 审计 P0-6.2）：
- 阈值真正成为定位失败条件，不只是 warning
- 使用 lane_width 动态判断横向误差
- 几何 fallback 使用空间索引
"""
import math
import logging
from typing import Optional, Tuple
from .schema import FineNode
from .store import RoadGraphStore

logger = logging.getLogger("LiteV1.RoadnetLocator")


class RoadPosition:
    """UGV 在路网中的定位结果"""
    def __init__(self, fine_node_id: str, decision_node_id: Optional[str],
                 lateral_error_m: float, heading_error_deg: float,
                 road_id: int, section_id: int, lane_id: int, s_m: float,
                 is_valid: bool = True):
        self.fine_node_id = fine_node_id
        self.decision_node_id = decision_node_id
        self.lateral_error_m = lateral_error_m
        self.heading_error_deg = heading_error_deg
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.s_m = s_m
        self.is_valid = is_valid


class RoadLocator:
    """将 UGV transform 投影到冻结路网"""

    def __init__(self, store: RoadGraphStore,
                 lane_half_width_tolerance_m: float = 0.75,
                 max_z_error_m: float = 2.0,
                 max_heading_error_deg: float = 60.0):
        self.store = store
        self.lane_half_width_tolerance_m = lane_half_width_tolerance_m
        self.max_z_error_m = max_z_error_m
        self.max_heading_error_deg = max_heading_error_deg

    def locate(self, carla_world, x: float, y: float, z: float = 0.5,
               yaw_deg: float = 0.0, check_heading: bool = False) -> Optional[RoadPosition]:
        """优先用 CARLA project_to_road，然后匹配到同语义 fine node.
        P1-12 B1: check_heading=True 时 heading_error 纳入 is_valid 判断（仅用于 DRIVING 后 relocalize）。
        STOP_OBSERVE 等场景传 check_heading=False，不重新 locate。"""
        import carla
        try:
            loc = carla.Location(x=x, y=y, z=z)
            wp = carla_world.get_map().get_waypoint(loc, project_to_road=True,
                                                     lane_type=carla.LaneType.Driving)
        except Exception as e:
            logger.warning(f"CARLA projection failed: {e}")
            wp = None

        if wp is None:
            return self._geometric_fallback(x, y, z, yaw_deg)

        # Find fine node matching same road/section/lane and closest s
        best_node = None
        best_d = float("inf")
        lane_nodes = self.store.get_lane_nodes(wp.road_id, wp.section_id, wp.lane_id)
        if not lane_nodes:
            return self._geometric_fallback(x, y, z, yaw_deg)

        for node in lane_nodes:
            d = abs(node.s_m - wp.s) + math.hypot(node.x - wp.transform.location.x,
                                                    node.y - wp.transform.location.y)
            if d < best_d:
                best_d = d
                best_node = node

        if best_node is None:
            return self._geometric_fallback(x, y, z, yaw_deg)

        lateral_error = math.hypot(best_node.x - x, best_node.y - y)
        # P0-fix (2026-07-21): use projected waypoint z as reference, not the raw input z.
        # The input z may be plan z=0.5 (ground level) while the actual road is elevated
        # (e.g. bridge/overpass z=10m). Using raw input z causes false z_error rejection.
        reference_z = wp.transform.location.z
        z_error = abs(best_node.z - reference_z)
        heading_error = abs(self._normalize_angle(best_node.yaw_deg - yaw_deg))

        # Dynamic threshold based on lane width (P0-6.2 fix)
        max_lateral = best_node.lane_width_m / 2 + self.lane_half_width_tolerance_m
        lateral_invalid = lateral_error > max_lateral or z_error > self.max_z_error_m
        heading_invalid = check_heading and heading_error > self.max_heading_error_deg
        if lateral_invalid or heading_invalid:
            logger.warning(f"Locator threshold exceeded: lateral={lateral_error:.1f}m (max {max_lateral:.1f}), "
                          f"z={z_error:.1f}m, heading={heading_error:.1f}deg")
            return RoadPosition(
                fine_node_id=best_node.node_id,
                decision_node_id=f"D:{best_node.node_id}" if f"D:{best_node.node_id}" in self.store.decision_nodes else None,
                lateral_error_m=lateral_error,
                heading_error_deg=heading_error,
                road_id=best_node.road_id,
                section_id=best_node.section_id,
                lane_id=best_node.lane_id,
                s_m=best_node.s_m,
                is_valid=False,
            )

        decision_node_id = f"D:{best_node.node_id}" if f"D:{best_node.node_id}" in self.store.decision_nodes else None
        return RoadPosition(
            fine_node_id=best_node.node_id,
            decision_node_id=decision_node_id,
            lateral_error_m=lateral_error,
            heading_error_deg=heading_error,
            road_id=best_node.road_id,
            section_id=best_node.section_id,
            lane_id=best_node.lane_id,
            s_m=best_node.s_m,
            is_valid=True,
        )

    def _geometric_fallback(self, x: float, y: float, z: float, yaw_deg: float,
                            check_heading: bool = False) -> Optional[RoadPosition]:
        """CARLA 投影失败时的兜底：最近 fine node + 严格阈值"""
        best_node = None
        best_d = float("inf")
        for node in self.store.nodes.values():
            d = math.hypot(node.x - x, node.y - y)
            if d < best_d:
                best_d = d
                best_node = node
        if best_node is None:
            return None
        max_lateral = best_node.lane_width_m / 2 + self.lane_half_width_tolerance_m
        if best_d > max_lateral:
            return RoadPosition(
                fine_node_id=best_node.node_id,
                decision_node_id=None,
                lateral_error_m=best_d,
                heading_error_deg=abs(self._normalize_angle(best_node.yaw_deg - yaw_deg)),
                road_id=best_node.road_id,
                section_id=best_node.section_id,
                lane_id=best_node.lane_id,
                s_m=best_node.s_m,
                is_valid=False,
            )
        heading_error = abs(self._normalize_angle(best_node.yaw_deg - yaw_deg))
        heading_ok = not check_heading or heading_error <= self.max_heading_error_deg
        return RoadPosition(
            fine_node_id=best_node.node_id,
            decision_node_id=f"D:{best_node.node_id}" if f"D:{best_node.node_id}" in self.store.decision_nodes else None,
            lateral_error_m=best_d,
            heading_error_deg=heading_error,
            road_id=best_node.road_id,
            section_id=best_node.section_id,
            lane_id=best_node.lane_id,
            s_m=best_node.s_m,
            is_valid=heading_ok,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def nearest_fine_node(self, x: float, y: float) -> Optional[FineNode]:
        best_node = None
        best_d = float("inf")
        for node in self.store.nodes.values():
            d = math.hypot(node.x - x, node.y - y)
            if d < best_d:
                best_d = d
                best_node = node
        return best_node
