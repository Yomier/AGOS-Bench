"""Directed public-graph coverage; geometry measures sampling, never visibility."""
from __future__ import annotations

import heapq
import math
from collections import deque

from agos.lite_core import LiteAction


def _get(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _xy(item):
    return float(_get(item, "x")), float(_get(item, "y"))


class CoveragePlanner:
    def __init__(self, services, public: dict, config: dict):
        self.services, self.public, self.config = services, public, config
        nav = services.semantic_nav
        if nav is None or not nav.port_owner or services.road_graph is None:
            raise ValueError("cooperative_search: public executable graph required")
        region = public["region"]
        self.xy = {}
        self.adj = {}
        for port in sorted(nav.port_owner):
            node = services.road_graph.nodes.get(port)
            if node is None:
                raise ValueError(f"cooperative_search: missing public position for port {port}")
            point = _xy(node)
            self.xy[port] = point
            options = nav.get_maneuver_options(
                port, intersection_max_distance_m=float(config.get("intersection_max_distance_m", 50)))
            self.adj[port] = [
                (str(opt.target_fine_node_id), float(opt.path_distance_m), str(action))
                for action, opt in sorted(options.items())
            ]
            if any(not math.isfinite(cost) or cost < 0 for _, cost, _ in self.adj[port]):
                raise ValueError("cooperative_search: nonfinite or negative route cost")
        if any(target not in self.adj for row in self.adj.values() for target, _, _ in row):
            raise ValueError("cooperative_search: executable edge leaves public port graph")
        self.units = {key: xy for key, xy in self.xy.items()
                      if region["x_min"] <= xy[0] <= region["x_max"]
                      and region["y_min"] <= xy[1] <= region["y_max"]}
        self.grid = {str(wp["id"]): (float(wp["x"]), float(wp["y"]))
                     for wp in public["uav_grid_waypoints"] if wp.get("active", True)}
        if not self.units or not self.grid:
            raise ValueError("cooperative_search: nonempty region units and active UAV grid required")
        altitude = float(public["start"]["uav"].get("z", 80))
        fov = float(config.get("uav_fov_deg", 90))
        self.radius = float(config.get("footprint_radius_m", altitude * math.tan(math.radians(fov / 2))))
        self.spacing = float(config.get("grid_spacing_m", 40))
        if not (math.isfinite(self.radius) and self.radius > 0 and self.spacing > 0):
            raise ValueError("cooperative_search: invalid sampling footprint/spacing")
        self.air_units = {key: {unit for unit, point in self.units.items()
                               if math.dist(point, station) <= self.radius}
                          for key, station in self.grid.items()}
        self.positions = {role.upper(): _xy(pose) for role, pose in public["start"].items()}
        self.samples = []
        self.assigned = {"UAV": set(), "UGV": set()}
        self.initial_counts = {"UAV": 0, "UGV": 0}
        self.version = 0
        self.joint = False
        self._dist_cache = {}
        self.last_status = {}

    def _nearest_port(self, point):
        return min(self.xy, key=lambda port: (math.dist(point, self.xy[port]), port))

    def _paths(self, start):
        if start in self._dist_cache:
            return self._dist_cache[start]
        dist, previous, queue = {start: 0.0}, {}, [(0.0, start)]
        while queue:
            distance, port = heapq.heappop(queue)
            if distance != dist.get(port):
                continue
            for target, cost, action in self.adj.get(port, ()):
                candidate = distance + cost
                if candidate < dist.get(target, math.inf):
                    dist[target] = candidate
                    previous[target] = (port, action)
                    heapq.heappush(queue, (candidate, target))
        self._dist_cache[start] = dist, previous
        return dist, previous

    def _completed(self, role, event_time):
        return set().union(*(units for sample_role, ready, units, _ in self.samples
                                 if sample_role == role and ready <= event_time))

    def _visited_stations(self, event_time):
        return {station for role, ready, _, station in self.samples
                if role == "UAV" and ready <= event_time and station is not None}

    def _snake(self, station_ids, start):
        variants = []
        for axis in (0, 1):
            bands = {}
            for key in station_ids:
                bands.setdefault(self.grid[key][axis], []).append(key)
            order = []
            for index, band in enumerate(sorted(bands)):
                order.extend(sorted(bands[band], key=lambda key: (self.grid[key][1 - axis], key),
                                    reverse=bool(index % 2)))
            for seq in (order, list(reversed(order))):
                cost = math.dist(start, self.grid[seq[0]]) if seq else 0
                cost += sum(math.dist(self.grid[a], self.grid[b]) for a, b in zip(seq, seq[1:]))
                variants.append((cost, seq))
        return min(variants)[1]

    def start(self, joint: bool) -> None:
        self.joint = bool(joint)
        self._allocate(0)
        self.initial_counts = {role: len(units) for role, units in self.assigned.items()}

    def _allocate(self, event_time):
        completed = self._completed("UAV", event_time) | self._completed("UGV", event_time)
        remaining = set(self.units) - completed
        ground_start = self._nearest_port(self.positions["UGV"])
        reachable, _ = self._paths(ground_start)
        aerial = set().union(*self.air_units.values())
        self.unreachable = sorted(remaining - (set(reachable) | aerial))
        if not self.joint:
            self.assigned = {"UGV": remaining.intersection(reachable),
                             "UAV": remaining.intersection(aerial)}
        else:
            assigned = {"UAV": set(), "UGV": set()}
            elapsed = {"UAV": 0.0, "UGV": 0.0}
            tail = {"UAV": self.positions["UAV"], "UGV": ground_start}
            speed = {"UAV": float(self.config.get("uav_speed_mps", 10)),
                     "UGV": float(self.config.get("ugv_speed_mps", 6))}
            if min(speed.values()) <= 0:
                raise ValueError("cooperative_search: positive public motion speeds required")
            # Air-only units first; remaining units use incremental directed travel
            # and the current min-max load. No reverse edge or asymmetric 2-opt.
            for unit in sorted(remaining, key=lambda key: (key in reachable, key)):
                costs = []
                if unit in aerial:
                    station = min((key for key, covered in self.air_units.items() if unit in covered),
                                  key=lambda key: (math.dist(tail["UAV"], self.grid[key]), key))
                    delta = math.dist(tail["UAV"], self.grid[station]) / speed["UAV"]
                    costs.append((max(elapsed["UAV"] + delta, elapsed["UGV"]),
                                  elapsed["UAV"] + delta, "UAV", self.grid[station]))
                distances, _ = self._paths(tail["UGV"])
                if unit in distances:
                    delta = distances[unit] / speed["UGV"]
                    costs.append((max(elapsed["UGV"] + delta, elapsed["UAV"]),
                                  elapsed["UGV"] + delta, "UGV", unit))
                if not costs:
                    if unit not in self.unreachable:
                        self.unreachable.append(unit)
                    continue
                _, load, role, endpoint = min(costs)
                assigned[role].add(unit)
                elapsed[role], tail[role] = load, endpoint
            self.assigned = assigned
        stations = [key for key, units in self.air_units.items()
                    if units.intersection(self.assigned["UAV"])]
        self.station_order = self._snake(stations, self.positions["UAV"])
        self.version += 1

    def observe(self, obs, executed_action, public_result: dict) -> None:
        if public_result.get("executed") is False or public_result.get("error"):
            return
        point = tuple(public_result.get("pos", obs.agent_position))[:2]
        role = obs.role
        self.positions[role] = point
        if public_result.get("sampled") is not True:
            return
        ready = float(public_result.get("ready_at", getattr(obs, "coop_event_time", 0)))
        if role == "UAV":
            station = min(self.grid, key=lambda key: (math.dist(point, self.grid[key]), key))
            covered = {unit for unit, xy in self.units.items() if math.dist(point, xy) <= self.radius}
            if math.dist(point, self.grid[station]) > 1.0:
                station = None
        else:
            station = None
            port = getattr(obs, "current_fine_node_id", None)
            if port is None or math.dist(point, self.xy.get(port, (math.inf, math.inf))) > 1.0:
                port = self._nearest_port(point)
            covered = {port} if port in self.units and math.dist(point, self.xy[port]) <= 1.0 else set()
        self.samples.append((role, ready, covered, station))

    def reallocate(self, reason: str, positions: dict, event_time: float) -> None:
        for role, point in positions.items():
            self.positions[role] = tuple(point)[:2]
        if self.joint:
            self._allocate(event_time)
        self.last_status = {"status": "reallocated" if self.joint else "fixed_resume", "reason": reason}

    def _air_next(self, obs, goal):
        # Expand through eligible local grid neighbors, then enforce the current
        # benchmark menu. A global station never becomes a teleport action.
        start = min(self.grid, key=lambda key: (math.dist(obs.agent_position[:2], self.grid[key]), key))
        previous, queue = {start: None}, deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            for target in sorted(self.grid):
                dx = abs(self.grid[node][0] - self.grid[target][0])
                dy = abs(self.grid[node][1] - self.grid[target][1])
                if target != node and dx <= self.spacing * 1.5 and dy <= self.spacing * 1.5:
                    if target not in previous:
                        previous[target] = node
                        queue.append(target)
        if goal not in previous or goal == start:
            return None
        target = goal
        while previous[target] != start:
            target = previous[target]
        legal = {_get(wp, "waypoint_id") for wp in obs.available_waypoints}
        if target not in legal or "MOVE_TO_UAV_WAYPOINT" not in obs.available_actions:
            return None
        return LiteAction("UAV", "MOVE_TO_UAV_WAYPOINT", waypoint_id=target)

    def next_action(self, obs):
        event_time = float(getattr(obs, "coop_event_time", 0))
        role = obs.role
        done = self._completed(role, event_time)
        if self.joint:
            done |= self._completed("UGV" if role == "UAV" else "UAV", event_time)
        remaining = self.assigned[role] - done
        if role == "UAV":
            visited = self._visited_stations(event_time)
            for station in self.station_order:
                if station in visited or not self.air_units[station].intersection(remaining):
                    continue
                if math.dist(obs.agent_position[:2], self.grid[station]) <= 1:
                    continue
                action = self._air_next(obs, station)
                if action:
                    self.last_status = {"status": "route_found", "goal": station}
                    return action
        else:
            current = getattr(obs, "current_fine_node_id", None)
            if current in self.adj:
                distances, previous = self._paths(current)
                candidates = [(distances[unit], unit) for unit in remaining
                              if unit in distances and unit != current]
                if candidates:
                    _, goal = min(candidates)
                    target = goal
                    while previous[target][0] != current:
                        target = previous[target][0]
                    action = previous[target][1]
                    if action in obs.available_actions:
                        self.last_status = {"status": "route_found", "goal": goal}
                        return LiteAction(role, action)
        self.last_status = {"status": "completed" if not remaining else "no_legal_progress",
                            "remaining_units": sorted(remaining)}
        rest = "WAIT" if role == "UAV" else "STOP_OBSERVE"
        if rest not in obs.available_actions:
            raise ValueError("cooperative_search: no legal coverage or observation action")
        return LiteAction(role, rest)

    def summary(self, role: str, event_time: float) -> dict:
        done = self._completed(role, event_time)
        return {"version": self.version, "role": role,
                "assigned_units": sorted(self.assigned[role]),
                "completed_units": sorted(done), "initial_count": self.initial_counts[role],
                "station_order": list(self.station_order) if role == "UAV" else [],
                "unreachable_units": sorted(self.unreachable),
                "sampling_proxy": "geometric_footprint_not_visibility"}
