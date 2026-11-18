from __future__ import annotations

import heapq

from typing import Any, Dict, Iterable, Optional, Tuple

class SparseRouteTool:
    """Compute a candidate-approach action on the executable sparse graph."""

    def __init__(self, intersection_max_distance_m: float = 50.0):
        self.intersection_max_distance_m = float(intersection_max_distance_m)
        self._cache_key = None
        self._adj: Dict[str, Dict[str, Tuple[str, float]]] = {}
        self._radj: Dict[str, list] = {}

    def _build_graph(self, adapter: Any) -> None:
        # B18（2026-09-17 审计）：缓存 key 在 id(adapter) 基础上加
        # len(adapter.port_owner)——同一 adapter 对象重锚定后 port 集数量变化时
        # 必须重建图，否则命中过期缓存（图内容已随 port_owner 改变）。
        key = (id(adapter), len(getattr(adapter, "port_owner", {}) or {}),
               getattr(adapter, "release_id", None), self.intersection_max_distance_m)
        if key == self._cache_key:
            return
        adj: Dict[str, Dict[str, Tuple[str, float]]] = {}
        radj: Dict[str, list] = {}
        ports = sorted(getattr(adapter, "port_owner", {}).keys())
        for port in ports:
            try:
                menu = adapter.get_maneuver_options(
                    port, intersection_max_distance_m=self.intersection_max_distance_m)
            except Exception:
                menu = {}
            row = {}
            for action, opt in sorted(menu.items()):
                target = str(getattr(opt, "target_fine_node_id", "") or "")
                if not target:
                    continue
                cost = max(0.0, float(getattr(opt, "path_distance_m", 0.0) or 0.0))
                row[str(action)] = (target, cost)
                radj.setdefault(target, []).append((port, cost, str(action)))
            adj[port] = row
        self._cache_key = key
        self._adj = adj
        self._radj = radj

    @staticmethod
    def _service_ids(active_candidate: Any) -> Tuple[Tuple[str, ...], str]:
        if active_candidate is None:
            return (), "none"
        if isinstance(active_candidate, dict):
            prefer = bool(active_candidate.get("prefer_confirmable_service", False))
            safe = active_candidate.get("confirmable_service_node_ids", ())
            broad = active_candidate.get("service_node_ids", ())
        else:
            prefer = bool(getattr(active_candidate, "prefer_confirmable_service", False))
            safe = getattr(active_candidate, "confirmable_service_node_ids", ())
            broad = getattr(active_candidate, "service_node_ids", ())
        ids = safe if (prefer and safe) else broad
        contract = "confirmable_farthest_corner" if (prefer and safe) else "broad_nearest_edge"
        return tuple(str(x) for x in (ids or ()) if x), contract

    def plan(self, adapter: Any, current_port: str, active_candidate: Any) -> Dict[str, Any]:
        out = {
            "activated": False,
            "status": "INACTIVE",
            "preferred_action": None,
            "remaining_m": None,
            "service_port_count": 0,
            "goal_contract": "none",
            "action_costs_m": {},
        }
        if adapter is None or not current_port or active_candidate is None:
            out["status"] = "MISSING_INPUT"
            return out
        self._build_graph(adapter)
        current_port = str(current_port)
        if current_port not in self._adj:
            out["status"] = "CURRENT_PORT_NOT_IN_SPARSE_GRAPH"
            return out

        _service_tuple, _goal_contract = self._service_ids(active_candidate)
        service_ids = set(_service_tuple)
        out["goal_contract"] = _goal_contract
        goals = sorted(service_ids.intersection(self._adj.keys()))
        out["service_port_count"] = len(goals)
        if not goals:
            out["status"] = "NO_EXECUTABLE_SERVICE_PORT"
            return out
        if current_port in service_ids:
            out.update({"activated": True, "status": "AT_SERVICE_PORT", "remaining_m": 0.0})
            return out

        # Multi-source reverse Dijkstra gives exact cost-to-go for every sparse port.
        dist = {g: 0.0 for g in goals}
        pq = [(0.0, g) for g in goals]
        heapq.heapify(pq)
        while pq:
            du, u = heapq.heappop(pq)
            if du != dist.get(u):
                continue
            for pred, edge_cost, _action in self._radj.get(u, ()):  # pred -> u
                nd = du + edge_cost
                if nd + 1e-9 < dist.get(pred, float("inf")):
                    dist[pred] = nd
                    heapq.heappush(pq, (nd, pred))

        if current_port not in dist:
            out["status"] = "SERVICE_PORT_UNREACHABLE"
            return out

        action_costs = {}
        best = None
        for action, (target, edge_cost) in self._adj.get(current_port, {}).items():
            if target not in dist:
                continue
            total = edge_cost + dist[target]
            action_costs[action] = round(total, 3)
            cand = (total, action)
            if best is None or cand < best:
                best = cand
        out["action_costs_m"] = dict(sorted(action_costs.items()))
        if best is None:
            out["status"] = "NO_LEGAL_PROGRESS_ACTION"
            return out
        out.update({
            "activated": True,
            "status": "ROUTE_FOUND",
            "preferred_action": best[1],
            "remaining_m": round(float(dist[current_port]), 3),
        })
        return out
