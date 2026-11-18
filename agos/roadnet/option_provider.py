"""roadnet/option_provider.py - 共同局部候选生成器 (v2)

关键改进（按外部 Agent 审计 P0-3.8）：
- Forward 10/30m 只沿 longitudinal corridor，不跨 junction entry
- Intersection options 直接读取预计算 junction movements
- Backward 10m 不用全图欧氏最近 anchor，用 route_history 上的实际点
"""
import math
import heapq
import logging
from typing import List, Dict, Optional, Tuple
from .schema import DecisionNode, DecisionEdge, FineNode, FineEdge
from .store import RoadGraphStore
from .router import RoadRouter, RouteFailure

logger = logging.getLogger("LiteV1.RoadnetOptionProvider")

LONGITUDINAL_EDGE_TYPES = {"LANE_FOLLOW", "JUNCTION_ENTER", "JUNCTION_INTERNAL",
                           "JUNCTION_EXIT", "MAP_NATIVE_UTURN"}


class RoadMoveOption:
    """UGV 可选择的局部道路动作"""
    def __init__(self, option_id: str, option_type: str,
                 target_decision_node_id: str, target_fine_node_id: str,
                 path_distance_m: float, maneuver: str, route_digest: str,
                 display_text: str, junction_id: Optional[int] = None,
                 fine_path: Optional[Tuple[str, ...]] = None,
                 decision_edge_id: Optional[str] = None):
        self.option_id = option_id
        self.option_type = option_type
        self.target_decision_node_id = target_decision_node_id
        self.target_fine_node_id = target_fine_node_id
        self.path_distance_m = path_distance_m
        self.maneuver = maneuver
        self.route_digest = route_digest
        self.display_text = display_text
        self.junction_id = junction_id
        self.fine_path = fine_path
        self.decision_edge_id = decision_edge_id


class RoadOptionProvider:
    """为地面车辆提供统一的局部道路候选

    R2-P1-03（2026-09-04 外审 + 用户裁定）：BACK_TO_PREVIOUS_ANCHOR 协议删除。
    v2 动作空间（get_maneuver_options）中回退语义由 REVERSE 承担；
    anchor 资产仅用于 runner 的锚点经过审计（anchor_visit_history），
    不再参与动作菜单生成。本类的 anchor_catalog 依赖已一并移除。
    """

    def __init__(self, store: RoadGraphStore):
        self.store = store
        self.router = RoadRouter(store)

    def get_options(self, current_fine_node_id: str,
                    route_history: List[Tuple[str, float]],
                    max_options: int = 8,
                    forward_distances: Tuple[float, ...] = (10.0, 30.0),
                    intersection_max_distance_m: float = 30.0,
                    consecutive_backtracks: int = 0,
                    max_consecutive_backtracks: int = 2) -> List[RoadMoveOption]:
        """v1 遗留接口（仅 adapter 兼容层使用；v2 评测走 get_maneuver_options）。"""
        if current_fine_node_id not in self.store.nodes:
            return []
        options = []

        # 1. Forward 10m / 30m (只沿 longitudinal corridor, target must be real decision node)
        options.extend(self._forward_options(current_fine_node_id, list(forward_distances)))

        # 2. Intersection options (读取预计算 junction movements)
        options.extend(self._intersection_options(current_fine_node_id, intersection_max_distance_m))

        # 3. 回退：R2-P1-03 删除 BACK_TO_PREVIOUS_ANCHOR（v2 由 REVERSE 承担）

        # Deduplicate by target decision node id
        seen = set()
        unique = []
        for opt in options:
            if opt.target_decision_node_id not in seen:
                seen.add(opt.target_decision_node_id)
                unique.append(opt)
        options = unique

        options.sort(key=lambda o: (self._priority(o), o.path_distance_m))
        return options[:max_options]

    def _priority(self, opt: RoadMoveOption) -> int:
        order = {
            "INTERSECTION_LEFT": 1,
            "INTERSECTION_RIGHT": 2,
            "INTERSECTION_STRAIGHT": 3,
            "INTERSECTION_UTURN": 4,
            "FORWARD_10M": 5,
            "FORWARD_30M": 6,
        }
        return order.get(opt.option_type, 7)

    def _forward_options(self, current_fine_node_id: str,
                         target_distances: List[float]) -> List[RoadMoveOption]:
        """沿 longitudinal corridor 前进，不跨 junction entry。
        P1-2: target must be a real decision node (CRUISE_ANCHOR / JUNCTION_ENTRY / etc.)
        P0-1: distance must be close to target_dist, not 6m masquerading as 10m
        If no real decision node exists within the distance window, don't generate the option."""
        options = []
        for target_dist in target_distances:
            best = self._walk_corridor_to_decision(current_fine_node_id, target_dist)
            if best:
                did, fine_id, dist, path = best
                label = "FORWARD_10M" if abs(target_dist - 10.0) < 0.1 else "FORWARD_30M"
                options.append(RoadMoveOption(
                    option_id=label,
                    option_type=label,
                    target_decision_node_id=did,
                    target_fine_node_id=fine_id,
                    path_distance_m=dist,
                    maneuver="FOLLOW",
                    route_digest=self._path_digest(path),
                    display_text=f"{dist:.1f}m ahead on road",
                    fine_path=tuple(path),
                ))
        return options

    def _walk_corridor_to_decision(self, start_id: str, target_dist: float,
                                    tolerance: float = 3.0
                                    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """沿 longitudinal corridor 前进，找到距离 target_dist 最近的 downstream decision node。
        P0-1: 不在距离目标 5m 前提前返回，而是走完整走廊后选最优。
        P1-2: 返回的 did 必须存在于 store.decision_nodes。
        Returns: (decision_node_id, fine_node_id, accumulated_dist, path) or None"""
        current = start_id
        accumulated = 0.0
        path = [start_id]
        visited = {start_id}
        best = None  # (abs_error, did, fine_id, dist, path)
        max_dist = target_dist + tolerance + 2.0  # 走到目标+容差+1个节点

        while accumulated < max_dist:
            successors = []
            for dst, edge_list in self.store.successors(current).items():
                for edge in edge_list:
                    if edge.edge_type not in LONGITUDINAL_EDGE_TYPES:
                        continue
                    if edge.edge_type == "JUNCTION_ENTER":
                        continue  # Don't enter junction in forward
                    successors.append((dst, edge))
            if not successors:
                break
            nxt_id, edge = min(successors, key=lambda x: x[1].length_m)
            if nxt_id in visited:
                break
            accumulated += edge.length_m
            path.append(nxt_id)
            visited.add(nxt_id)
            current = nxt_id

            # Check if this fine node is a real decision node
            did = f"D:{nxt_id}"
            if did in self.store.decision_nodes:
                err = abs(accumulated - target_dist)
                if err <= tolerance:
                    if best is None or err < best[0]:
                        best = (err, did, nxt_id, accumulated, list(path))
                # If we've passed the target and found a decision node, stop
                if accumulated >= target_dist:
                    break

        if best is not None:
            _, did, fine_id, dist, best_path = best
            return did, fine_id, dist, best_path
        return None

    def _intersection_options(self, current_fine_node_id: str,
                              max_distance_m: float = 30.0) -> List[RoadMoveOption]:
        """查找前方最近的 junction entry，读取预计算 junction movements。
        P0-2: 允许当前点就是 JUNCTION_ENTRY（去掉 current != current_fine_node_id 排除）"""
        entry_candidates = []
        visited = {current_fine_node_id: 0.0}
        queue = [(0.0, current_fine_node_id, [current_fine_node_id])]
        while queue:
            dist, current, path = heapq.heappop(queue)
            if dist > max_distance_m:
                continue
            did = f"D:{current}"
            d_node = self.store.decision_nodes.get(did)
            # P0-2: allow current node to be junction entry (removed current != current_fine_node_id)
            if d_node and "JUNCTION_ENTRY" in d_node.roles:
                entry_candidates.append((did, current, dist, path))
                break  # Only need nearest
            for dst, edge_list in self.store.successors(current).items():
                for edge in edge_list:
                    if edge.edge_type not in LONGITUDINAL_EDGE_TYPES:
                        continue
                    if dst in path:
                        continue
                    new_dist = dist + edge.length_m
                    if new_dist < visited.get(dst, float("inf")):
                        visited[dst] = new_dist
                        heapq.heappush(queue, (new_dist, dst, path + [dst]))

        if not entry_candidates:
            return []

        entry_did, entry_fine_id, entry_dist, entry_path = entry_candidates[0]
        # Read pre-computed junction movements from decision graph
        movements = self.store.decision_successors(entry_did)
        options = []
        for exit_did, edge_list in movements.items():
            for d_edge in edge_list:
                maneuver = d_edge.maneuver
                junction_path = list(d_edge.fine_path or ())
                # P0 (r5 TRACE regression review): a TURN option may be offered
                # while the UGV is still up to ~30m before the junction. The
                # precomputed DecisionEdge path starts at the junction entry,
                # whereas runner executes RoadMoveOption.fine_path as the full
                # authoritative path from the *current* node. Supplying only
                # d_edge.fine_path therefore skips/teleports over current->entry
                # while still charging the full entry_dist + junction distance.
                if not junction_path or junction_path[0] != entry_fine_id:
                    logger.warning(
                        "Invalid junction decision edge %s: expected fine_path to start at %s, got %s",
                        d_edge.decision_edge_id, entry_fine_id, junction_path[0] if junction_path else None)
                    continue
                full_path = list(entry_path)
                if not full_path or full_path[-1] != entry_fine_id:
                    logger.warning("Invalid entry path for junction %s from %s", entry_did, current_fine_node_id)
                    continue
                full_path.extend(junction_path[1:])  # avoid duplicate entry node
                total_dist = entry_dist + d_edge.path_length_m
                options.append(RoadMoveOption(
                    option_id=f"INTERSECTION_{maneuver}",
                    option_type=f"INTERSECTION_{maneuver}",
                    target_decision_node_id=exit_did,
                    target_fine_node_id=self.store.decision_nodes[exit_did].anchor_fine_node_id,
                    path_distance_m=total_dist,
                    maneuver=maneuver,
                    route_digest=d_edge.decision_edge_id,
                    display_text=f"{total_dist:.1f}m: {maneuver.lower()} at junction",
                    junction_id=d_edge.junction_id,
                    fine_path=tuple(full_path),
                    decision_edge_id=d_edge.decision_edge_id,
                ))
        return options

    def _path_digest(self, path: List[str]) -> str:
        import hashlib
        return hashlib.sha256(">".join(path).encode()).hexdigest()[:16]

    # ============================================================
    # 2026-09-04 UGV 动作空间 v2（用户裁定）：操纵式动作菜单
    # ============================================================
    def get_maneuver_options(self, current_fine_node_id: str,
                             route_history: List[Tuple[str, float]],
                             intersection_max_distance_m: float = 30.0,
                             consecutive_backtracks: int = 0,
                             max_consecutive_backtracks: int = 2
                             ) -> Dict[str, "RoadMoveOption"]:
        """返回 {action_name: RoadMoveOption} 操纵式动作菜单。

        - FORWARD_1: 沿道路前进 1 个 fine 节点（~10m，不进路口）
        - FORWARD_2: 前进 2 个 fine 节点（~20m；走廊中断则该档不可用）
        - TURN_LEFT / TURN_RIGHT: 前方最近路口（含当前即路口节点）的左/右转
        - REVERSE: 原路倒车回退 1 个 fine 节点（防震荡限制沿用）
        """
        menu: Dict[str, RoadMoveOption] = {}

        # fine node 间距约 2m：按目标距离行进（FORWARD_1≈10m、FORWARD_2≈20m），
        # 不跨 junction entry（路口附近由 TURN_* / REVERSE 承担）。
        interior = self._is_junction_interior(current_fine_node_id)
        for target_dist, act_name, zh_desc in ((10.0, "FORWARD_1", "advance ~10m"),
                                                (20.0, "FORWARD_2", "advance ~20m")):
            walked = self._forward_distance_walk(current_fine_node_id, target_dist)
            if walked is None and interior:
                # 路口内部逃生（F0 法医 2026-09-07）：UGV 被生成器出生在路口内部时，
                # 普通 FORWARD 走廊（严格 LANE_FOLLOW）与 TURN（entry 在身后、
                # BFS 不可达）、REVERSE（无行驶历史）三路全断 → 菜单恒空、
                # UGV 永久锁死（mid_00004/00005、hard_00001 全方法 0m 实锤）。
                # 此处"驶出路口"不违反 R2-P1-01——该规则禁止的是从入口跳过
                # 转向决策穿路口，而路口内部已越过决策点，唯一合理前进是驶出。
                walked = self._junction_exit_walk(current_fine_node_id, target_dist)
            if walked is None:
                continue
            path, dist = walked
            target = path[-1]
            menu[act_name] = RoadMoveOption(
                option_id=act_name, option_type=act_name,
                target_decision_node_id=f"D:{target}",
                target_fine_node_id=target,
                path_distance_m=dist,
                maneuver="FOLLOW",
                route_digest=self._path_digest(path),
                display_text=zh_desc,
                fine_path=tuple(path),
            )

        for opt in self._intersection_options(current_fine_node_id, intersection_max_distance_m):
            if opt.maneuver == "LEFT":
                menu["TURN_LEFT"] = opt
            elif opt.maneuver == "RIGHT":
                menu["TURN_RIGHT"] = opt
            elif opt.maneuver == "STRAIGHT":
                # R2-P1-02（用户裁定）：v2 补路口直行——FORWARD_* 已严禁穿路口（P1-01），
                # 直行穿越必须显式选择；UTURN 不提供（REVERSE 承担回头语义）。
                menu["TURN_STRAIGHT"] = opt

        rev = self._reverse_node_option(current_fine_node_id, route_history,
                                        consecutive_backtracks, max_consecutive_backtracks)
        if rev is not None:
            menu["REVERSE"] = rev
        elif not menu:
            # 逃生门（2026-09-04 冒烟发现）：前方断头路且不在路口时菜单为空，
            # 若防震荡同时封死 REVERSE，UGV 将被锁死只能 STOP 到 budget 耗尽。
            # 此时绕过防震荡上限强制提供一次倒车脱困选项。
            rev = self._reverse_node_option(current_fine_node_id, route_history, 0, 1)
            if rev is not None:
                menu["REVERSE"] = rev

        return menu

    def _forward_distance_walk(self, current_fine_node_id: str, target_dist: float):
        """沿 longitudinal corridor 前进至累计距离 >= target_dist。

        R2-P1-01 收紧（外部二轮审查 + 用户裁定 2026-09-04）：
        - FORWARD_* 只允许 LANE_FOLLOW 边——遇任何 JUNCTION_*（ENTER/INTERNAL/EXIT）
          或 UTURN 边即停止，路口内动作一律由 TURN_LEFT/TURN_RIGHT/TURN_STRAIGHT
          显式承担，FORWARD 不再隐式穿越路口（实测 Town03 46% 的 forward 选项
          曾含 junction 边，语义污染严重）。
        - 距离容差：累计距离不足目标 80% 时不提供该档（防止 4m 短路径伪装成
          "约 20m" 的 FORWARD_2，扭曲动作代价与模型策略）。
        Returns: (path, total_dist) or None"""
        current = current_fine_node_id
        path = [current]
        total = 0.0
        while total < target_dist:
            successors = []
            for dst, edge_list in self.store.successors(current).items():
                for edge in edge_list:
                    if edge.edge_type != "LANE_FOLLOW":
                        continue  # R2-P1-01: 严格 LANE_FOLLOW，任何路口/UTURN 边即断走廊
                    successors.append((dst, edge))
            if not successors:
                break  # 走廊中断（路口入口/断头/无边）
            nxt_id, edge = min(successors, key=lambda x: x[1].length_m)
            if nxt_id in path:
                break
            total += edge.length_m
            path.append(nxt_id)
            current = nxt_id
        if len(path) < 2 or total < target_dist * 0.8:
            return None  # R2-P1-01: 距离不足 80% 不提供该档
        return path, total

    def _is_junction_interior(self, node_id: str) -> bool:
        """判断 fine node 是否位于路口内部（已越过 entry、尚未驶出到普通车道）。

        F0 法医（2026-09-07）：episode 生成器可能把 UGV 出生点放在路口
        内部/边界（mid_00004/00005、hard_00001 实锤）。判定规则：
        - 后继含 JUNCTION_INTERNAL / JUNCTION_EXIT 边（本节点在路口内或
          紧贴出口），且不含 JUNCTION_ENTER 边（ENTER 边只出现在路口
          入口节点上——入口节点必须保持 R2-P1-01 语义，不得借此穿路口）。
        """
        if node_id not in self.store.nodes:
            return False
        has_enter = False
        has_internal_or_exit = False
        for _dst, edge_list in self.store.successors(node_id).items():
            for edge in edge_list:
                if edge.edge_type == "JUNCTION_ENTER":
                    has_enter = True
                elif edge.edge_type in ("JUNCTION_INTERNAL", "JUNCTION_EXIT"):
                    has_internal_or_exit = True
        return has_internal_or_exit and not has_enter

    def _junction_exit_walk(self, current_fine_node_id: str, target_dist: float):
        """路口内部逃生走廊：沿 INTERNAL/EXIT 边驶出路口，过 EXIT 边后转
        LANE_FOLLOW 继续前进至目标距离。

        仅当 FORWARD 严格走廊返回 None 且 _is_junction_interior 为真时由
        get_maneuver_options 调用（F0 法医 2026-09-07：出生在路口内部的
        UGV 三路全断——FORWARD 严格 LANE_FOLLOW、TURN 找不到身后的 entry、
        REVERSE 无历史——菜单恒空导致永久锁死）。

        语义边界：驶出路口 ≠ 违反 R2-P1-01。P1-01 禁止的是从路口入口
        跳过转向决策穿路口；路口内部已越过决策点，唯一合理前进就是驶出。
        复用 _forward_distance_walk 的 80% 距离规则与防环规则。

        Returns: (path, total_dist) or None"""
        current = current_fine_node_id
        path = [current]
        total = 0.0
        visited = {current}
        exited = False  # 是否已越过 JUNCTION_EXIT 边
        while total < target_dist:
            successors = []
            for dst, edge_list in self.store.successors(current).items():
                for edge in edge_list:
                    if not exited:
                        # 路口内：只沿 INTERNAL/EXIT 边前进
                        if edge.edge_type in ("JUNCTION_INTERNAL", "JUNCTION_EXIT"):
                            successors.append((dst, edge))
                    else:
                        # 已驶出：回归严格 LANE_FOLLOW
                        if edge.edge_type == "LANE_FOLLOW":
                            successors.append((dst, edge))
            if not successors:
                break  # 无出口的内部死角（罕见）——不伪造选项
            nxt_id, edge = min(successors, key=lambda x: x[1].length_m)
            if nxt_id in visited:
                break
            if edge.edge_type == "JUNCTION_EXIT":
                exited = True
            total += edge.length_m
            path.append(nxt_id)
            visited.add(nxt_id)
            current = nxt_id
        if len(path) < 2 or total < target_dist * 0.8:
            return None  # 与 FORWARD 相同的 80% 距离规则
        return path, total

    def _reverse_node_option(self, current_fine_node_id: str,
                             route_history: List[Tuple[str, float]],
                             consecutive_backtracks: int,
                             max_consecutive_backtracks: int) -> Optional[RoadMoveOption]:
        """原地倒车：沿 route_history 原路回退（物理倒挡），目标回退距离约 10m
        （fine node 间距约 2m，即回退约 5 个节点；历史不足则退到历史起点）。

        防震荡：连续回退 >= max_consecutive_backtracks 次后不再提供。
        """
        if consecutive_backtracks >= max_consecutive_backtracks:
            return None
        hist_ids = [h[0] for h in route_history]
        if current_fine_node_id not in hist_ids or len(hist_ids) < 2:
            return None
        end_i = len(hist_ids) - 1 - hist_ids[::-1].index(current_fine_node_id)
        if end_i <= 0:
            return None
        # 回溯目标：累计里程回退 >= 8m 的最近历史节点（至少回退 1 个节点）
        start_odom = route_history[end_i][1]
        back_i = end_i - 1
        while back_i > 0 and (start_odom - route_history[back_i][1]) < 8.0:
            back_i -= 1
        prev_fine = hist_ids[back_i]
        if prev_fine == current_fine_node_id:
            return None
        seg = hist_ids[back_i:end_i + 1]
        fine_path = tuple(reversed(seg))  # (current, ..., prev) 物理倒挡路径
        dist = abs(start_odom - route_history[back_i][1])
        return RoadMoveOption(
            option_id="REVERSE", option_type="REVERSE_NODE",
            target_decision_node_id=f"D:{prev_fine}",
            target_fine_node_id=prev_fine,
            path_distance_m=dist,
            maneuver="BACKTRACK",
            route_digest=self._path_digest(list(fine_path)),
            display_text=f"reverse {dist:.1f}m straight back",
            fine_path=fine_path,
        )
