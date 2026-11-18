#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantic_nav_adapter.py —— sparse_nav_v1 语义动作适配器（外审四轮 P0-BENCH 落地）。

在 v4_1 锚点网络（L2 控制图）上实现操纵式六动作，与 runner 现役
UGV_MANEUVER_ACTIONS（FORWARD_1/FORWARD_2/TURN_LEFT/TURN_RIGHT/TURN_STRAIGHT/REVERSE）
同键、同 option_type 白名单，SFT collector 与评测 runner 共用本实现（差分测试正本）。

动作语义（docs/designs/2026-09-11-sparse-nav-runner-design.md，用户 2026-09-11 拍板）：
  FORWARD_1      前进：与参考方向同向（夹角<45°）的出边，1 跳（~30m）
  FORWARD_2      大幅前进：连续 2 次 FORWARD_1 合成一步，中间节点 pass-through
  REVERSE        倒车：正后方（夹角>135°）最近的入边，沿其 fine_path 反序执行，
                 车头朝向锁定为参考方向（yaw_override_deg）
  TURN_LEFT/RIGHT/STRAIGHT  路口动作：沿前进链累计 ≤intersection_max_distance_m（默认 50m）
                 内存到 junction_in（pass-through）→ IN 的对应方向出边（IN→OUT）
                 → OUT 首条出边到新道路首节点（用户拍板：落点=OUT 后新道路首节点）
  方向分类：出边方向相对进入方向 左=+45..135° / 右=-135..-45° / 直=±45°
  参考方向：同向出边（<45°）圆均值；无同向出边时全部出边圆均值

执行契约：resolved.fine_path 为多边拼接的权威 L0 节点序列，runner 沿其逐节点
teleport（E3b 已验证机制），path_distance_m = 各边 distance_m 之和（真实路径长度）。
状态推进以最后一条边的 to_port_fine_node 为权威；禁止 CARLA get_waypoint 反推。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ACTION_SPACE_VERSION = "sparse_nav_v1"

# 与 runner option_type 白名单一致（runner.py 执行直走分支）
OPTION_TYPE = {
    "FORWARD_1": "FORWARD_1",
    "FORWARD_2": "FORWARD_2",
    "REVERSE": "REVERSE_NODE",
    "TURN_LEFT": "INTERSECTION_LEFT",
    "TURN_RIGHT": "INTERSECTION_RIGHT",
    "TURN_STRAIGHT": "INTERSECTION_STRAIGHT",
}


class SemanticNavError(RuntimeError):
    """所有拒绝信号（非法状态/动作、菜单缺失）。"""


class SemanticNavOption:
    __slots__ = ("action", "option_type", "edge_ids", "fine_path", "path_distance_m",
                 "target_port", "target_control_node_id", "maneuver_text",
                 "yaw_override_deg", "pass_through_ports", "aliases", "state_semantic_type")

    def __init__(self, action, option_type, edge_ids, fine_path, path_distance_m,
                 target_port, target_control_node_id, maneuver_text,
                 yaw_override_deg=None, pass_through_ports=(),
                 aliases=(), state_semantic_type=""):
        self.action = action
        self.option_type = option_type
        self.edge_ids = tuple(edge_ids)
        self.fine_path = tuple(fine_path)
        self.path_distance_m = path_distance_m
        self.target_port = target_port
        self.target_control_node_id = target_control_node_id
        self.maneuver_text = maneuver_text
        self.yaw_override_deg = yaw_override_deg
        self.pass_through_ports = tuple(pass_through_ports)
        # 外审 R9 P1-05 方案A（2026-09-13 用户拍板）：同执行别名去重后，
        # 被隐藏的同 (target_port, fine_path) 动作名记录于此（可追溯，不进菜单）。
        self.aliases = tuple(aliases)
        # 外审 R9 P1-06：状态语义类型（junction_in / chain），供 SFT schema 分层。
        self.state_semantic_type = state_semantic_type

    @property
    def target_fine_node_id(self):
        """runner MANEUVER_V2 分发/菜单的鸭子类型兼容别名（RoadMoveOption 同名字段）。"""
        return self.target_port

    @property
    def display_text(self):
        return self.maneuver_text

    def as_action_menu_item(self):
        """runner 菜单项（兼容 _render_maneuver_menu 四键 + sparse provenance）。"""
        return {
            "action": self.action,
            "description": self.maneuver_text,
            "distance_m": round(self.path_distance_m, 1),
            "target_xy": None,                     # 由 runner 注入（图坐标查询）
            "target_fine_node_id": self.target_port,
            "option_type": self.option_type,
            "edge_ids": list(self.edge_ids),
            "action_space_version": ACTION_SPACE_VERSION,
        }


def _norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _circ_mean(angles):
    mx = sum(math.cos(a) for a in angles) / len(angles)
    my = sum(math.sin(a) for a in angles) / len(angles)
    return math.atan2(my, mx)


# 外审 R9 P1-05 方案A（2026-09-13 用户拍板）：同执行别名去重优先级。
# TURN_* > FORWARD_2 > FORWARD_1 > REVERSE——路口宏语义优先于前进链合成。
_ALIAS_PRIORITY = {
    "TURN_LEFT": 0, "TURN_RIGHT": 0, "TURN_STRAIGHT": 0,
    "FORWARD_2": 1, "FORWARD_1": 2, "REVERSE": 3,
}


class SemanticNavAdapter:
    """六语义动作 → L2 边序列解析。组合控制图索引（与 SparseWaypointAdapter 同源加载）。"""

    def __init__(self, payload):
        self.graph = payload
        self.release_id = payload.get("control_graph_release_id")
        self.schema_version = payload.get("schema_version")
        self.nodes = {n["control_node_id"]: n for n in payload["control_nodes"]}
        self.node_type = {n["control_node_id"]: n["type"] for n in payload["control_nodes"]}
        self.port_owner = {}
        for n in payload["control_nodes"]:
            for p in n.get("ports", []):
                self.port_owner[p] = n["control_node_id"]
        self.by_from_port = defaultdict(list)
        self.by_to_port = defaultdict(list)
        for e in payload.get("edge_variants", []):
            self.by_from_port[e["from_port_fine_node"]].append(e)
            self.by_to_port[e["to_port_fine_node"]].append(e)
        for k in self.by_from_port:
            self.by_from_port[k].sort(key=lambda e: (e["distance_m"], e["edge_id"]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    # ---------- 基础查询 ----------

    def control_node_for_port(self, port):
        owner = self.port_owner.get(port)
        if owner is None:
            raise SemanticNavError(f"port {port} is not a control port of this graph")
        return owner

    def _pos(self, node_id):
        n = self.nodes[node_id]
        return n["x"], n["y"]

    def _edge_dir(self, e):
        (ax, ay), (bx, by) = self._pos(e["from_control_node_id"]), self._pos(e["to_control_node_id"])
        if abs(bx - ax) < 1e-6 and abs(by - ay) < 1e-6:
            return None
        return math.atan2(by - ay, bx - ax)

    def _forward_edge(self, port):
        """该 port 的前进边：与参考方向同向（<45°）且夹角最小。

        2026-09-12 E5a 三轮修复（语义死锁）：出边方向不可判定
        （_edge_dir 返回 None——同控制节点自环微边两端 cn 坐标相同）
        但**出边唯一**时，该边即前进（唯一去向无方向歧义）。
        此前这类 port 的 F1 被静默排除，菜单只剩 REVERSE，
        形成 F1→REV 死循环（Town10HD e5a_00005 实证 s7-s60）。"""
        outs = self.by_from_port.get(port, [])
        if not outs:
            return None
        angs = [self._edge_dir(e) for e in outs]
        angs = [a for a in angs if a is not None]
        if not angs:
            return outs[0] if len(outs) == 1 else None
        ref = _circ_mean(angs)
        best, best_d = None, 1e9
        for e in outs:
            a = self._edge_dir(e)
            if a is None:
                continue
            d = abs(_norm_angle(a - ref))
            if d < math.radians(45) and d < best_d:
                best, best_d = e, d
        return best

    def _ref_dir(self, port):
        for cand in (self.by_from_port.get(port, []), self.by_to_port.get(port, [])):
            angs = [self._edge_dir(e) for e in cand]
            angs = [a for a in angs if a is not None]
            if angs:
                return _circ_mean(angs)
        return None

    # ---------- 动作组装 ----------

    @staticmethod
    def _join_fine_path(edges, enter_reverse=False):
        """多边 fine_path 拼接：后续边去掉与前一尾重复的首节点；倒车整体反序。"""
        fp = []
        for e in edges:
            seg = e["fine_path"]
            if not fp:
                fp.extend(seg)
            else:
                start = 0
                if seg and seg[0] == fp[-1]:
                    start = 1
                fp.extend(seg[start:])
        if enter_reverse:
            fp.reverse()
        return fp

    def _chain_option(self, action, edges, enter_reverse=False, yaw_override=None):
        fp = self._join_fine_path(edges, enter_reverse)
        dist = sum(e["distance_m"] for e in edges)
        last_from = edges[0] if enter_reverse else edges[-1]
        target_node = last_from["from_control_node_id"] if enter_reverse else last_from["to_control_node_id"]
        target_port = last_from["from_port_fine_node"] if enter_reverse else last_from["to_port_fine_node"]
        pt = [] if enter_reverse else [e["to_port_fine_node"] for e in edges[:-1]]
        return SemanticNavOption(
            action=action, option_type=OPTION_TYPE[action],
            edge_ids=[e["edge_id"] for e in edges], fine_path=fp,
            path_distance_m=round(dist, 3), target_port=target_port,
            target_control_node_id=target_node,
            maneuver_text=f"{action} -> {target_node} ({dist:.0f}m, {len(edges)} edges)",
            yaw_override_deg=yaw_override, pass_through_ports=pt)

    # ---------- 路口动作 ----------

    def _intersection_options(self, arrive_chain, in_port=None):
        """arrive_chain=到达 IN 的前进边序列；in_port 为空 chain 时的显式 IN port。

        进入方向：末边方向；空 chain 时用该 IN 入边圆均值（无历史妥协）。
        落点=OUT 后首条出边终点（新道路首节点）；OUT 无出边时落点=OUT 本身。
        """
        if in_port is None:
            if not arrive_chain:
                return {}
            in_port = arrive_chain[-1]["to_port_fine_node"]
        enter_dir = self._edge_dir(arrive_chain[-1]) if arrive_chain else None
        if enter_dir is None:
            angs = [self._edge_dir(e) for e in self.by_to_port.get(in_port, [])]
            angs = [a for a in angs if a is not None]
            if not angs:
                return {}
            enter_dir = _circ_mean(angs)
        cands = {"TURN_LEFT": (None, 1e9), "TURN_RIGHT": (None, 1e9), "TURN_STRAIGHT": (None, 1e9)}
        for e in self.by_from_port.get(in_port, []):
            # 转向方向判据：OUT 之后新道路的行进方向（OUT 的前进出边 nxt），而非
            # IN→OUT 的 16-24m 短连线（该连线被路口整体偏移主导，实测 rel 全落 ±46°
            # 不可分辨）。与"落点=OUT 后新道路首节点"的拍板语义天然一致：方向与落点
            # 用同一条 nxt，保证分类与执行统一。
            nxt = self._forward_edge(e["to_port_fine_node"])
            a = self._edge_dir(nxt) if nxt is not None else self._edge_dir(e)
            if a is None:
                continue
            # CARLA 左手系（y 轴向下）：几何左转 = enter - 出路方向 为正（旋向取反）
            d = math.degrees(_norm_angle(enter_dir - a))
            key = None
            if 45 <= d <= 135:
                key = "TURN_LEFT"
            elif -135 <= d <= -45:
                key = "TURN_RIGHT"
            elif abs(d) <= 45:
                key = "TURN_STRAIGHT"
            if key is None:
                continue
            closeness = abs(d - (90 if key == "TURN_LEFT" else -90 if key == "TURN_RIGHT" else 0))
            if closeness < cands[key][1]:
                cands[key] = (e, closeness)
        out = {}
        for key, (e, _) in cands.items():
            if e is None:
                continue
            segs = list(arrive_chain) + [e]
            nxt = self._forward_edge(e["to_port_fine_node"])
            if nxt is not None:
                segs.append(nxt)
            out[key] = self._chain_option(key, segs)
        return out

    # ---------- 主入口 ----------

    def get_maneuver_options(self, current_port, intersection_max_distance_m=50.0):
        """返回 {action_key: SemanticNavOption}；空 dict=无可用动作（dead port）。"""
        if current_port not in self.port_owner:
            raise SemanticNavError(f"port {current_port} is not a control port of this graph")
        menu = {}
        fwd = self._forward_edge(current_port)
        ref = self._ref_dir(current_port)

        if fwd is not None:
            menu["FORWARD_1"] = self._chain_option("FORWARD_1", [fwd])
            f2 = self._forward_edge(fwd["to_port_fine_node"])
            if f2 is not None:
                menu["FORWARD_2"] = self._chain_option("FORWARD_2", [fwd, f2])

        # REVERSE：from 端位于参考方向正后方（夹角>135°）且最近的入边
        if ref is not None:
            cx, cy = self._pos(self.port_owner[current_port])
            best, best_d = None, 1e9
            for e in self.by_to_port.get(current_port, []):
                fx, fy = self._pos(e["from_control_node_id"])
                if abs(fx - cx) < 1e-6 and abs(fy - cy) < 1e-6:
                    continue
                back_angle = math.atan2(fy - cy, fx - cx)
                if abs(_norm_angle(back_angle - ref)) > math.radians(135):
                    d = math.hypot(cx - fx, cy - fy)
                    if d < best_d:
                        best, best_d = e, d
            if best is not None:
                menu["REVERSE"] = self._chain_option(
                    "REVERSE", [best], enter_reverse=True,
                    yaw_override=round(math.degrees(ref), 1))

        # 路口动作：当前已在 IN → 直接分类；否则沿前进链累计 ≤max 找 IN
        if self.node_type.get(self.port_owner.get(current_port)) == "junction_in":
            menu.update(self._intersection_options([], in_port=current_port))
        elif fwd is not None:
            chain, dist = [fwd], fwd["distance_m"]
            while dist <= intersection_max_distance_m:
                head = chain[-1]["to_port_fine_node"]
                if self.node_type.get(self.port_owner.get(head)) == "junction_in":
                    break
                nxt = self._forward_edge(head)
                if nxt is None or dist + nxt["distance_m"] > intersection_max_distance_m + 1e-6:
                    break
                chain.append(nxt)
                dist += nxt["distance_m"]
            head = chain[-1]["to_port_fine_node"]
            if self.node_type.get(self.port_owner.get(head)) == "junction_in":
                menu.update(self._intersection_options(chain))

        # 外审 R9 P1-05 方案A（2026-09-13 用户拍板）：same-execution 别名去重。
        # key=(target_port, fine_path)；保留优先级最高动作，其余记入 aliases 不进菜单。
        # P1-06：全部选项标注 state_semantic_type（junction_in / chain）供 SFT 分层。
        state_sem = "junction_in" if self.node_type.get(
            self.port_owner.get(current_port)) == "junction_in" else "chain"
        for opt in menu.values():
            opt.state_semantic_type = state_sem
        groups = {}
        for name, opt in menu.items():
            groups.setdefault((opt.target_port, tuple(opt.fine_path)), []).append(name)
        if all(len(v) == 1 for v in groups.values()):
            return menu
        deduped = {}
        for names in groups.values():
            names_sorted = sorted(names, key=lambda n: _ALIAS_PRIORITY.get(n, 99))
            keep = names_sorted[0]
            opt = menu[keep]
            if len(names_sorted) > 1:
                opt.aliases = tuple(sorted(names_sorted[1:]))
            deduped[keep] = opt
        return deduped

    def resolve(self, current_port, action, intersection_max_distance_m=50.0):
        menu = self.get_maneuver_options(current_port, intersection_max_distance_m)
        opt = menu.get(action)
        if opt is None:
            raise SemanticNavError(
                f"action {action} not available at {current_port} (menu={sorted(menu)})")
        return opt
