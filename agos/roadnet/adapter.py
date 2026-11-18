"""roadnet/adapter.py - 旧接口兼容适配器 (v2)

兼容旧 UGVRoadCatalog.get_available 和 select_approach_waypoint 接口，
底层使用 v2 的 RoadOptionProvider 和 CandidateApproachPlanner。
"""
from typing import List, Dict, Tuple, Optional
from .option_provider import RoadOptionProvider
from .candidate_approach import CandidateApproachPlanner
from .router import RoadRouter, RouteFailure
from agos.benchmark_protocol import DEFAULT_CONFIRM_DISTANCE_M


class UGVRoadCatalogAdapter:
    """兼容旧 UGVRoadCatalog.get_available 接口"""

    def __init__(self, option_provider: RoadOptionProvider,
                 current_fine_node_id: str,
                 route_history: List[Tuple[str, float]],
                 current_yaw: float = 0.0,
                 consecutive_backtracks: int = 0):
        self.option_provider = option_provider
        self.current_fine_node_id = current_fine_node_id
        self.route_history = route_history
        self.current_yaw = current_yaw
        self.consecutive_backtracks = consecutive_backtracks

    def get_available(self, forward_opts=(10, 30), intersection_max=30,
                      max_consecutive_backtracks=2):
        options = self.option_provider.get_options(
            self.current_fine_node_id,
            self.route_history,
            max_options=8,
            forward_distances=tuple(forward_opts),
            intersection_max_distance_m=intersection_max,
            consecutive_backtracks=self.consecutive_backtracks,
            max_consecutive_backtracks=max_consecutive_backtracks,
        )
        wps = []
        # 2026-07-24: maneuver 英文转中文
        maneuver_map = {
            "FOLLOW": "直行",
            "STRAIGHT": "路口直行",
            "LEFT": "左转",
            "RIGHT": "右转",
            "UTURN": "掉头",
        }
        for opt in options:
            node = self.option_provider.store.nodes.get(opt.target_fine_node_id)
            if node is None:
                continue
            maneuver_zh = maneuver_map.get(opt.maneuver, opt.maneuver)
            display_zh = f"{maneuver_zh}({opt.path_distance_m:.0f}m)"
            wps.append({
                "waypoint_id": opt.target_fine_node_id,
                "decision_node_id": opt.target_decision_node_id,
                "world_xy": (node.x, node.y),
                "distance_m": round(opt.path_distance_m, 1),
                "description": display_zh,
                "maneuver": maneuver_zh,
                "option_id": opt.option_id,
                "option_type": opt.option_type,
                "visited": False,
                "fine_path": list(opt.fine_path) if opt.fine_path else None,
            })
        return wps


class ApproachAdapter:
    """兼容旧 select_approach_waypoint 接口"""

    def __init__(self, approach_planner: CandidateApproachPlanner,
                 current_fine_node_id: str,
                 require_confirmable_goal: bool = False):
        self.approach_planner = approach_planner
        self.current_fine_node_id = current_fine_node_id
        # B10（2026-09-17 审计）：实例级默认值，select() 可被调用参数覆盖（OR 语义）。
        # runner 侧由 ugv.approach_require_confirmable config 传入，默认 False=旧行为。
        self.require_confirmable_goal = bool(require_confirmable_goal)

    def check_candidate_serviceability(self,
                                       candidate_polygon: List[Tuple[float, float]],
                                       max_confirm_distance_m: float = DEFAULT_CONFIRM_DISTANCE_M,
                                       preferred_min_m: float = 30.0,
                                       preferred_max_m: float = 50.0,
                                       max_route_m: float = 300.0) -> Tuple[bool, str, int]:
        """检查 candidate polygon 是否可被 UGV 服务（CandidateServiceContract，外审 P0-C01 修复）。

        基于 service pool：先 compute_service_nodes（半径 = max_confirm_distance_m，
        与 runner 创建点 / approach planner 三处共用同一池），再 pool_reachable 判
        当前节点到池是否存在有界可达路径。池内无任何可达节点 → 打回标注（外部行为不变）。

        Returns:
            (serviceable, reason, inside_node_count)
            - serviceable: True 表示池中至少存在一个可达 approach node
            - reason: 描述信息（含池大小，如 "pool=12 ring_ok"）
            - inside_node_count: 池内位于 polygon 内部（最近边距离=0）的节点数
        """
        from .candidate_approach import (
            compute_service_nodes, pool_reachable, _polygon_distance,
        )

        if self.current_fine_node_id not in self.approach_planner.store.nodes:
            return False, "CURRENT_NOT_IN_GRAPH", 0
        if len(candidate_polygon) < 3:
            return False, "INVALID_POLYGON", 0

        store = self.approach_planner.store
        pool = compute_service_nodes(store, candidate_polygon,
                                     service_radius_m=max_confirm_distance_m)
        if not pool:
            return False, "CANDIDATE_NO_REACHABLE_NODE_INSIDE", 0
        if not pool_reachable(store, self.current_fine_node_id, pool,
                              max_route_m=max_route_m):
            return False, "CANDIDATE_NO_REACHABLE_NODE_INSIDE", 0

        inside_count = 0
        for nid in pool:
            node = store.nodes[nid]
            if _polygon_distance(node.x, node.y, candidate_polygon) == 0.0:
                inside_count += 1
        if inside_count > 0:
            return True, f"REACHABLE_INSIDE pool={len(pool)}", inside_count
        return True, f"REACHABLE_RING pool={len(pool)} ring_ok", 0

    def select(self, candidate_region_center_xy: Tuple[float, float],
               candidate_polygon: List[Tuple[float, float]],
               min_dist: float = 30, max_dist: float = 50,
               fallback_min: float = 25, fallback_max: float = 60,
               max_route_m: float = 300.0,
               max_confirm_distance_m: float = DEFAULT_CONFIRM_DISTANCE_M,
               require_confirmable_goal: bool = False):
        """P1-6: 传参给 approach_planner. P0-4: 使用 bounded Dijkstra.
        距离契约：fallback_max 不超过 max_confirm_distance_m，否则 CONFIRM 必然失败。
        bounded Dijkstra 找不到合格点时返回 None（not serviceable），不 fallback 到 nearest。

        B10（2026-09-17 审计）：require_confirmable_goal 透传给 planner.plan()
        （与构造函数实例级默认 OR 语义，任一为 True 即生效）。
        """
        # 距离契约：fallback_max 不能超过 max_confirm_distance_m
        effective_fallback_max = min(fallback_max, max_confirm_distance_m)
        effective_max_dist = min(max_dist, max_confirm_distance_m)

        route, err = self.approach_planner.plan(
            candidate_polygon, self.current_fine_node_id,
            preferred_min_m=min_dist, preferred_max_m=effective_max_dist,
            fallback_min_m=fallback_min, fallback_max_m=effective_fallback_max,
            max_route_m=max_route_m,
            require_confirmable_goal=bool(require_confirmable_goal or self.require_confirmable_goal),
        )
        if route is None:
            # 不再 fallback 到 nearest_reachable_node - 候选不可达即标记 not serviceable
            return None, 0, err or "CANDIDATE_NOT_SERVICEABLE_FROM_ROAD"

        last_node_id = route.node_path[-1]
        last_node = self.approach_planner.store.nodes[last_node_id]
        return {
            "id": last_node_id,
            "x": last_node.x,
            "y": last_node.y,
            "z": last_node.z,
            "yaw": last_node.yaw_deg,
        }, route.path_length_m, "roadnet"
