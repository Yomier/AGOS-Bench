"""roadnet/candidate_approach.py - 疑似目标区域的道路接近规划 (v3)

v3 核心改进（2026-07-21 用户要求）：
- 两级都没有时返回 CANDIDATE_NO_REACHABLE_NODE_INSIDE，由 runner 打回 UAV 重新标注
"""
import math
import heapq
import logging
from typing import List, Tuple, Optional
from .router import RoutePlan, RouteFailure, RoadRouter
from .store import RoadGraphStore
from agos.benchmark_protocol import DEFAULT_CONFIRM_DISTANCE_M

logger = logging.getLogger("LiteV1.CandidateApproach")

MAX_APPROACH_DISTANCE_M = 60.0
MAX_CONFIRM_DISTANCE_M = DEFAULT_CONFIRM_DISTANCE_M
ALPHA = 0.3  # polygon_distance weight in score


def _point_in_polygon(x, y, polygon: List[Tuple[float, float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    projx = x1 + t * dx
    projy = y1 + t * dy
    return math.hypot(px - projx, py - projy)


def _polygon_distance(px, py, polygon: List[Tuple[float, float]]) -> float:
    if _point_in_polygon(px, py, polygon):
        return 0.0
    min_d = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < min_d:
            min_d = d
    return min_d


def _polygon_max_distance(px, py, polygon: List[Tuple[float, float]]) -> float:
    """Distance from point to the FARTHEST polygon corner.

    P0-fix: if this is <= max_confirm_distance_m, then ANY point inside the
    polygon (including the true target wherever it sits in the cell) is within
    confirm range of (px, py). This is the conservative guarantee the runtime
    needs — measuring to the NEAREST polygon edge lets a far-corner target
    exceed the confirm threshold.
    """
    return max(math.hypot(px - vx, py - vy) for vx, vy in polygon)


# ============================================================
# CandidateServiceContract（外审 P0-C01 修复，Phase-G2 Stage 0 Step G1）
# ============================================================
def compute_service_nodes(store: RoadGraphStore,
                          candidate_polygon: List[Tuple[float, float]],
                          service_radius_m: float) -> frozenset:
    """候选的服务节点池：所有到 polygon 最近边距离 <= service_radius_m 的 fine 节点。

    位置无关（纯几何，不含可达性），供三处共用同一池（CandidateServiceContract）：
      1) adapter.check_candidate_serviceability 的可服务性判定；
      2) runner 候选创建点把池固化进 Candidate.service_node_ids；
      3) runner 到达判定（池成员即到达，取代旧"距中心 <= approach_distance_min_m"）。
    plan() 的三级池（inside / preferred 30-50m / near <= max_confirm_distance_m）
    本身全部落在 poly_dist <= max_confirm_distance_m 内，故半径与
    max_confirm_distance_m 同源时 plan() 选出的 goal 必在池内。
    """
    pool = set()
    for node_id, node in store.nodes.items():
        if _polygon_distance(node.x, node.y, candidate_polygon) <= service_radius_m:
            pool.add(node_id)
    return frozenset(pool)


def compute_confirmable_service_nodes(store: RoadGraphStore,
                                       candidate_polygon: List[Tuple[float, float]],
                                       confirm_radius_m: float) -> frozenset:
    """Conservative verification-ready node pool.

    A node is admitted only when the farthest candidate-polygon corner is within
    confirm_radius_m.  This guarantees the distance part of terminal confirmation
    for any target point inside the candidate polygon.  It does NOT guarantee LOS
    or visual detectability; those remain runtime terminal checks.

    This pool is intentionally separate from compute_service_nodes(), whose
    nearest-edge <= radius contract is useful for broad candidate serviceability
    but is too weak to call an arrival "verification-ready".
    """
    pool = set()
    for node_id, node in store.nodes.items():
        if _polygon_max_distance(node.x, node.y, candidate_polygon) <= float(confirm_radius_m):
            pool.add(node_id)
    return frozenset(pool)


# ============================================================
# GSAM 两实验（2026-10-10 设计 §5/§9.1）：点类型候选服务池
# ============================================================
def compute_point_service_nodes(store: RoadGraphStore,
                                point_xy: Tuple[float, float],
                                bands_m=(20.0, 35.0, 50.0),
                                min_m: float = 5.0,
                                max_band_m: float = DEFAULT_CONFIRM_DISTANCE_M) -> Tuple[frozenset, Optional[float]]:
    """点候选的三级可达观察 port 池（开发默认 5-20 / 5-35 / 5-50m，冻结后不可改）。

    与 region 池同源（store.nodes 纯几何，不含可达性；可达性由 runner 的
    sparse_service_pool_reachable 预检补足）。返回 (pool, band_used)：
    逐级取第一个非空档；全空返回 (frozenset(), None)（调用方标记 unserviceable）。
    min_m 下限保证不在目标点上（车辆占据空间，需在观察点核验）；
    最终档兜底 max_band_m 与 max_confirm_distance_m 同源（当前30m；显式50m为历史协议）。
    """
    px, py = float(point_xy[0]), float(point_xy[1])
    band_used = None
    for band in bands_m or (20.0, 35.0, 50.0):
        band = min(float(band), float(max_band_m))
        pool = frozenset(
            node_id for node_id, node in store.nodes.items()
            if min_m <= math.hypot(node.x - px, node.y - py) <= band)
        if pool:
            return pool, band
    return frozenset(), None


def point_diagnostic_square(point_xy: Tuple[float, float],
                            half_side_m: float) -> List[Tuple[float, float]]:
    """点候选的评估诊断方形（仅用于 grid_hit/ACG 等评估指标与地图显示，
    不参与服务池/路由/到达合同——设计 §7.3 禁止以构造 polygon 掩盖接口缺失）。"""
    px, py = float(point_xy[0]), float(point_xy[1])
    h = max(1.0, float(half_side_m))
    return [(px - h, py - h), (px + h, py - h), (px + h, py + h), (px - h, py + h)]


def pool_reachable(store: RoadGraphStore,
                   current_fine_node_id: str,
                   service_node_ids,
                   max_route_m: Optional[float] = None) -> bool:
    """从 current 做有界 Dijkstra（逻辑复用 plan()），存在池内可达节点即返回 True。

    max_route_m=None（或 <=0 / inf）表示无界。current 不在图中或池为空时返回 False。
    """
    pool = set(service_node_ids or ())
    if not pool:
        return False
    if current_fine_node_id not in store.nodes:
        return False
    _unbounded = (max_route_m is None) or (max_route_m <= 0) or (max_route_m == float("inf"))
    dist_map: dict = {current_fine_node_id: 0.0}
    pq = [(0.0, current_fine_node_id)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist_map[u]:
            continue
        if not _unbounded and d > max_route_m:
            continue
        if u in pool:
            return True
        for v, edge_list in store.successors(u).items():
            for e in edge_list:
                nd = d + e.length_m
                if (_unbounded or nd <= max_route_m) and nd < dist_map.get(v, float("inf")):
                    dist_map[v] = nd
                    heapq.heappush(pq, (nd, v))
    return False


class CandidateApproachPlanner:
    """从冻结 2m 图为 candidate polygon 选择可达接近点"""

    def __init__(self, store: RoadGraphStore,
                 max_approach_distance_m: float = MAX_APPROACH_DISTANCE_M,
                 max_confirm_distance_m: float = MAX_CONFIRM_DISTANCE_M):
        self.store = store
        self.router = RoadRouter(store)
        self.max_approach_distance_m = max_approach_distance_m
        self.max_confirm_distance_m = max_confirm_distance_m

    def plan(self, candidate_polygon: List[Tuple[float, float]],
             current_fine_node_id: str,
             max_distance_m: float = MAX_APPROACH_DISTANCE_M,
             preferred_min_m: float = 30.0,
             preferred_max_m: float = 50.0,
             fallback_min_m: float = 25.0,
             fallback_max_m: float = 60.0,
             max_route_m: float = 300.0,
             require_confirmable_goal: bool = False,
             ) -> Tuple[Optional[RoutePlan], Optional[str]]:

        """三级策略：polygon 内部可达点 > 外部 30-50m 环 > 外部 25-60m 环。

        P1-8 (2026-09-04 external review, doc fix): 筛选按"到 polygon 最近边距离"
        分层（与 adapter.check_candidate_serviceability 一致），并非保守的
        "最远角 <= max_confirm" 保证——max_corner_dist 仅用于 inside 池的打分加权。
        选中节点不构成"certified confirmable"承诺，仅为 approach node
        （到达后仍需 STOP_OBSERVE 满足距离/LOS/可见性成功条件）。

        B10（2026-09-17 审计）：require_confirmable_goal=True 时 preferred/near 池
        仅保留 _polygon_max_distance(node) <= max_confirm_distance_m（服务半径，
        当前默认30m）的节点可作为 goal——把上述"非保守保证"升级为保守保证
        （goal 到 polygon 任意点都在确认距离内）。默认 False=完全旧行为。
        runner 侧由 ugv.approach_require_confirmable config 传入。

        Returns:
            (RoutePlan, None) on success
            (None, error_code) on failure
        """
        if current_fine_node_id not in self.store.nodes:
            return None, "CURRENT_NOT_IN_GRAPH"
        if len(candidate_polygon) < 3:
            return None, "INVALID_POLYGON"

        # Direct callers must obey the same cap as ApproachAdapter. A stale
        # preferred 30-50m annulus must not select an out-of-pool goal at 30m.
        preferred_max_m = min(float(preferred_max_m), self.max_confirm_distance_m)

        # 1. Bounded Dijkstra from current.
        _unbounded = (max_route_m is None) or (max_route_m <= 0) or (max_route_m == float("inf"))
        dist_map: dict = {current_fine_node_id: 0.0}
        pq = [(0.0, current_fine_node_id)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist_map[u]:
                continue
            if not _unbounded and d > max_route_m:
                continue
            for v, edge_list in self.store.successors(u).items():
                for e in edge_list:
                    nd = d + e.length_m
                    if (_unbounded or nd <= max_route_m) and nd < dist_map.get(v, float("inf")):
                        dist_map[v] = nd
                        heapq.heappush(pq, (nd, v))

        # 2026-07-25 修复：统一用"到 polygon 最近边距离"判断，不再区分 large_poly。
        # 与 adapter.check_candidate_serviceability 保持一致。
        preferred_mid = (preferred_min_m + preferred_max_m) / 2.0
        inside_candidates = []
        preferred_candidates = []
        near_candidates = []

        for node_id, route_dist in dist_map.items():
            node = self.store.nodes[node_id]
            poly_dist = _polygon_distance(node.x, node.y, candidate_polygon)
            max_corner_dist = _polygon_max_distance(node.x, node.y, candidate_polygon)

            if poly_dist == 0.0:
                # R22 外审 P1 NAV-02：inside 池与 preferred/near 同受保守最远角约束——
                # 特大 polygon 内部节点到最远角可 >50m。注意该过滤仅证明距离可确认，
                # 不保证目标在 polygon 内、LOS 通畅或可识别。
                if require_confirmable_goal and max_corner_dist > self.max_confirm_distance_m:
                    continue
                score = route_dist + ALPHA * max_corner_dist
                inside_candidates.append((score, node_id, route_dist, poly_dist, max_corner_dist))
            elif preferred_min_m <= poly_dist <= preferred_max_m:
                # B10：要求可确认 goal 时，最远角超服务半径的节点不入 preferred 池
                if require_confirmable_goal and max_corner_dist > self.max_confirm_distance_m:
                    continue
                score = route_dist + ALPHA * abs(poly_dist - preferred_mid)
                preferred_candidates.append((score, node_id, route_dist, poly_dist, max_corner_dist))
            elif poly_dist <= self.max_confirm_distance_m:
                # B10：同上，near 池过滤超远角节点
                if require_confirmable_goal and max_corner_dist > self.max_confirm_distance_m:
                    continue
                score = route_dist + ALPHA * poly_dist
                near_candidates.append((score, node_id, route_dist, poly_dist, max_corner_dist))

        # 3. 按优先级选候选池
        if inside_candidates:
            candidates = inside_candidates
            pool = "inside"
        elif preferred_candidates:
            candidates = preferred_candidates
            pool = "preferred_ring"
        elif near_candidates:
            candidates = near_candidates
            pool = "near_edge"
        else:
            return None, "CANDIDATE_NO_REACHABLE_NODE_INSIDE"

        # 4. 选最优 score
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]
        best_node_id = best[1]
        logger.debug(
            "Approach pool=%s, best_node=%s, route_dist=%.1f, poly_dist=%.1f, max_corner=%.1f",
            pool, best_node_id, best[2], best[3], best[4]
        )

        # 5. Plan route to best node
        res = self.router.plan(current_fine_node_id, best_node_id, cost_mode="distance")
        if isinstance(res, RouteFailure):
            return None, "ROUTE_PLAN_FAILED"
        return res, None

    def nearest_reachable_node(self, xy: Tuple[float, float],
                               current_fine_node_id: str,
                               max_distance_m: float = MAX_APPROACH_DISTANCE_M
                               ) -> Tuple[Optional[str], Optional[str]]:
        best_node = None
        best_d = float("inf")
        for node in self.store.nodes.values():
            d = math.hypot(node.x - xy[0], node.y - xy[1])
            if d < best_d and d <= max_distance_m:
                best_d = d
                best_node = node
        if best_node is None:
            return None, "NO_NEAR_ROAD_NODE"
        res = self.router.plan(current_fine_node_id, best_node.node_id, cost_mode="distance")
        if isinstance(res, RouteFailure):
            return None, res.code
        return best_node.node_id, None
