"""roadnet/router.py - 有向 A* 路由 (v2)

关键修复（按外部 Agent 审计 P0-3.9）：
- time 模式 heuristic 量纲修正
- RoutePlan 独立累计 path_length_m 和 estimated_time_s
- 支持多边（multigraph）
"""
import heapq
import math
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .schema import FineNode, FineEdge
from .store import RoadGraphStore

logger = logging.getLogger("LiteV1.RoadnetRouter")

MAX_SPEED_MPS = 5.0


@dataclass
class RoutePlan:
    node_path: Tuple[str, ...]
    edge_path: Tuple[str, ...]
    path_length_m: float
    estimated_time_s: float
    turn_count: int
    route_digest: str


@dataclass
class RouteFailure:
    code: str
    start: str
    goal: str
    graph_id: str


class RoadRouter:
    """基于冻结有向图的路由器（支持 multigraph）"""

    def __init__(self, store: RoadGraphStore):
        self.store = store

    def plan(self, start_fine_node_id: str, goal_fine_node_id: str,
             cost_mode: str = "distance") -> RoutePlan | RouteFailure:
        if start_fine_node_id not in self.store.nodes:
            return RouteFailure(code="START_NOT_IN_GRAPH", start=start_fine_node_id,
                               goal=goal_fine_node_id, graph_id=self.store.manifest.graph_id)
        if goal_fine_node_id not in self.store.nodes:
            return RouteFailure(code="GOAL_NOT_IN_GRAPH", start=start_fine_node_id,
                               goal=goal_fine_node_id, graph_id=self.store.manifest.graph_id)
        if start_fine_node_id == goal_fine_node_id:
            return RoutePlan(
                node_path=(start_fine_node_id,),
                edge_path=(),
                path_length_m=0.0,
                estimated_time_s=0.0,
                turn_count=0,
                route_digest=self._digest([start_fine_node_id], []),
            )

        came_from: Dict[str, str] = {}
        edge_used: Dict[str, str] = {}
        g_score: Dict[str, float] = {start_fine_node_id: 0.0}
        f_score: Dict[str, float] = {start_fine_node_id: self._heuristic(start_fine_node_id, goal_fine_node_id, cost_mode)}
        open_set = [(f_score[start_fine_node_id], start_fine_node_id)]
        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == goal_fine_node_id:
                return self._reconstruct(current, came_from, edge_used, g_score)
            closed.add(current)

            for dst, edge_list in self.store.successors(current).items():
                if dst in closed:
                    continue
                # Use first edge in multigraph (shortest for cost)
                best_edge = min(edge_list, key=lambda e: e.length_m if cost_mode == "distance" else e.base_travel_time_s)
                cost = best_edge.length_m if cost_mode == "distance" else best_edge.base_travel_time_s
                tentative_g = g_score.get(current, float("inf")) + cost
                if tentative_g < g_score.get(dst, float("inf")):
                    came_from[dst] = current
                    edge_used[dst] = best_edge.edge_id
                    g_score[dst] = tentative_g
                    f_score[dst] = tentative_g + self._heuristic(dst, goal_fine_node_id, cost_mode)
                    heapq.heappush(open_set, (f_score[dst], dst))

        return RouteFailure(code="NO_DIRECTED_PATH", start=start_fine_node_id,
                           goal=goal_fine_node_id, graph_id=self.store.manifest.graph_id)

    def _heuristic(self, node_id: str, goal_id: str, cost_mode: str) -> float:
        node = self.store.nodes[node_id]
        goal = self.store.nodes[goal_id]
        euclidean_m = math.hypot(node.x - goal.x, node.y - goal.y)
        if cost_mode == "time":
            return euclidean_m / MAX_SPEED_MPS
        return euclidean_m

    def _reconstruct(self, current: str, came_from: Dict[str, str],
                     edge_used: Dict[str, str], g_score: Dict[str, float]) -> RoutePlan:
        path = [current]
        edges = []
        while current in came_from:
            parent = came_from[current]
            edges.append(edge_used[current])
            path.append(parent)
            current = parent
        path.reverse()
        edges.reverse()

        # Independent accumulation (P0-3.9 fix)
        total_length = 0.0
        total_time = 0.0
        turn_count = 0
        for eid in edges:
            edge = self.store.edges[eid]
            total_length += edge.length_m
            total_time += edge.base_travel_time_s
            if edge.turn_hint in ("LEFT", "RIGHT", "UTURN"):
                turn_count += 1

        return RoutePlan(
            node_path=tuple(path),
            edge_path=tuple(edges),
            path_length_m=total_length,
            estimated_time_s=total_time,
            turn_count=turn_count,
            route_digest=self._digest(path, edges),
        )

    def _digest(self, path: List[str], edges: List[str]) -> str:
        content = ">".join(path) + "|" + ">".join(edges)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def plan_xy(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float],
                cost_mode: str = "distance") -> RoutePlan | RouteFailure:
        start_node = self._nearest_node(start_xy)
        goal_node = self._nearest_node(goal_xy)
        if start_node is None:
            return RouteFailure(code="START_NO_NEAR_NODE", start=str(start_xy),
                               goal=str(goal_xy), graph_id=self.store.manifest.graph_id)
        if goal_node is None:
            return RouteFailure(code="GOAL_NO_NEAR_NODE", start=str(start_xy),
                               goal=str(goal_xy), graph_id=self.store.manifest.graph_id)
        return self.plan(start_node.node_id, goal_node.node_id, cost_mode)

    def _nearest_node(self, xy: Tuple[float, float]) -> Optional[FineNode]:
        best = None
        best_d = float("inf")
        for node in self.store.nodes.values():
            d = math.hypot(node.x - xy[0], node.y - xy[1])
            if d < best_d:
                best_d = d
                best = node
        return best
