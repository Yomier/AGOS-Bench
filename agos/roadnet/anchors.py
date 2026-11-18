# -*- coding: utf-8 -*-
"""roadnet/anchors.py - UGV 锚点目录与生成器 (v3.1)

锚点体系（用户裁定规则）：
- 路口锚：center（观察位）+ 每方位 1 个 entry（同方位合并；被合并 arm 若失锚则补偿）；
  超大路口（对角 > big_junc_diag）补质心中点锚。
- 路段锚：方向粒度链（双向路两侧各留链、同向多车道合并为代表链），
  链内锚距目标 s_target=20m，修边 [gap_min, gap_max]=[15,25]；
  空档硬上限优先（L∈(25,30) 段保底放 1 锚，间隔 12.5~15m）。
- 路口内部锚：JUNCTION_INTERNAL 车道链 > gap_max 等分补 path_mid。
- fill 锚：带重锚定的 multi-source Dijkstra 兜底——沿任何可行驶路径相邻锚间距
  <= gap_max（+2m 节点量化），消除复合立交区锚沙漠。

每个锚点都是有向的：记录坐标 (x,y,z)、航向 yaw_deg、八方位 dir。
Town03 的 v3.1 路网共 647 个锚点；按行驶距离核验，最大间距为 26.7 m。
"""
import heapq
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# 全局过滤优先级：center/mid 先放，entry 次之，road_mid 再次，path_mid/fill 兜底让位
_PRIO = {"junction_center": 0, "junction_mid": 0, "junction_entry": 1,
         "junction_entry_arm": 1, "road_mid": 2, "junction_path_mid": 3}


def bearing_from_yaw(yaw_deg: float) -> str:
    """CARLA yaw（度）→ 八方位（N=+X 北）。

    项目坐标系约定（AGENTS.md）：+X=北、+Y=东；CARLA/UE yaw 从 +X 起顺时针增长
    （yaw=90 朝 +Y=东），因此 yaw 本身即罗盘角：北=0、东=90、南=180、西=270。
    2026-09-04 修正：旧公式 (90-yaw)%360 是数学角转罗盘，用于 yaw 上产生
    N↔E 镜像（仅污染锚名后缀与 dir 字段，分组/位置/间距不受影响）。
    """
    b = yaw_deg % 360.0
    return COMPASS[int((b + 22.5) // 45) % 8]


def bearing_from_vector(dx_north: float, dy_east: float) -> str:
    """位移向量 → 八方位（R2-P1-04 统一入口）。

    项目坐标系：+X=北、+Y=东。atan2(dy_east, dx_north) 即罗盘角
    （北=0、东=90、南=180、西=270），与 bearing_from_yaw 同口径。
    所有"两点连线定方位"的计算必须走本函数，禁止再写 90-θ 旧换算。
    """
    b = math.degrees(math.atan2(dy_east, dx_north)) % 360.0
    return COMPASS[int((b + 22.5) // 45) % 8]


@dataclass
class AnchorRecord:
    name: str
    type: str
    fine_node: str
    x: float
    y: float
    z: float
    yaw_deg: float
    dir: str
    junction_id: Optional[int] = None
    section_key: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["section_key"] = list(self.section_key) if self.section_key else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "AnchorRecord":
        d = dict(d)
        if d.get("section_key") is not None:
            d["section_key"] = tuple(d["section_key"])
        return cls(**d)


class AnchorCatalog:
    """UGV 锚点目录：冻结资产加载 + fine_node 索引。

    用途（R2-P1-03 定界，2026-09-04）：
    - runner 锚点经过检测（anchor_visit_history，轨迹审计/复盘用）
    - 不参与 v2 动作菜单决策：回退语义已由 REVERSE 承担，
      旧 BACK_TO_PREVIOUS_ANCHOR 协议已删除（详见 option_provider）
    """

    def __init__(self, anchors: List[AnchorRecord], graph_id: str = "",
                 params: Optional[Dict] = None, source_path: str = ""):
        self.anchors = anchors
        self.graph_id = graph_id
        self.params = params or {}
        self.source_path = source_path
        self.schema_version = None
        self.anchor_version = None
        self.by_fine_node: Dict[str, AnchorRecord] = {}
        for a in anchors:
            if a.fine_node not in self.by_fine_node:   # 同 fine_node 多锚取第一个
                self.by_fine_node[a.fine_node] = a

    def __len__(self) -> int:
        return len(self.anchors)

    @classmethod
    def load(cls, path: str) -> "AnchorCatalog":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        anchors = [AnchorRecord.from_dict(d) for d in payload.get("anchors", [])]
        cat = cls(anchors, graph_id=payload.get("graph_id", ""),
                  params=payload.get("params", {}), source_path=path)
        cat.schema_version = payload.get("schema_version")
        cat.anchor_version = payload.get("anchor_version")
        return cat

    @classmethod
    def load_for_graph_dir(cls, graph_dir: str, filename: str = "anchors_v3_1.json"
                           ) -> Optional["AnchorCatalog"]:
        """从路网图目录加载锚点资产；不存在则返回 None（该图暂无锚点，无回退选项）。"""
        path = os.path.join(graph_dir, filename)
        if not os.path.exists(path):
            return None
        return cls.load(path)


def save_catalog(anchors: List[AnchorRecord], path: str, graph_id: str = "",
                 params: Optional[Dict] = None,
                 source_manifest_sha256: str = "") -> None:
    """R2-P1-04：资产自带版本与溯源元数据。

    schema_version: 目录结构版本；anchor_version: 锚点协议版本（v3.1）；
    builder_version: 生成器逻辑版本（含方位公式修正）；
    source_manifest_sha256: 来源路网 manifest 的 asset_sha256（可校验
    "路网冻结了但锚点没跟着冻结"的复现漏洞）。
    """
    import datetime
    payload = {
        "schema_version": 2,
        "anchor_version": "v3.1",
        "builder_version": "anchors_v3_1_r2_20260904",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "graph_id": graph_id,
        "source_manifest_sha256": source_manifest_sha256,
        "params": params or {},
        "count": len(anchors),
        "anchors": [a.to_dict() for a in anchors],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def build_anchor_catalog(store,
                         s_target: float = 20.0,
                         gap_min: float = 15.0,
                         gap_max: float = 25.0,
                         min_global: float = 15.0,
                         big_junc_diag: float = 60.0,
                         min_junc_nodes: int = 4) -> List[AnchorRecord]:
    """在 store（RoadGraphStore）上构建 v3.1 锚点列表。

    Returns:
        List[AnchorRecord]（已按类型+名称排序，经过全局过滤与 fill 兜底）
    """
    nodes = store.nodes
    anchors: List[AnchorRecord] = []

    def add_anchor(name, atype, nid, jid=None, sec_key=None):
        n = nodes[nid]
        anchors.append(AnchorRecord(
            name=name, type=atype, fine_node=nid,
            x=round(n.x, 2), y=round(n.y, 2), z=round(n.z, 2),
            yaw_deg=round(n.yaw_deg, 1), dir=bearing_from_yaw(n.yaw_deg),
            junction_id=jid, section_key=sec_key))

    # ---------------- 边分类索引 ----------------
    follow_adj: Dict[str, List[str]] = defaultdict(list)
    follow_in: Dict[str, int] = defaultdict(int)
    follow_out: Dict[str, int] = defaultdict(int)
    int_adj: Dict[str, List[str]] = defaultdict(list)
    int_in: Dict[str, int] = defaultdict(int)
    int_out: Dict[str, int] = defaultdict(int)
    enter_by_junc: Dict[int, List[str]] = defaultdict(list)
    for e in store.edges.values():
        t = e.edge_type
        if t == "LANE_FOLLOW":
            follow_adj[e.src].append(e.dst)
            follow_out[e.src] += 1
            follow_in[e.dst] += 1
        elif t == "JUNCTION_INTERNAL":
            int_adj[e.src].append(e.dst)
            int_out[e.src] += 1
            int_in[e.dst] += 1
        elif t == "JUNCTION_ENTER" and e.junction_id is not None:
            enter_by_junc[e.junction_id].append(e.src)

    # ---------------- A. 路口锚 ----------------
    junc_nodes: Dict[int, List[str]] = defaultdict(list)
    for nid, n in nodes.items():
        if n.is_junction and n.junction_id is not None:
            junc_nodes[n.junction_id].append(nid)

    for jid in sorted(junc_nodes):
        members = junc_nodes[jid]
        if len(members) < min_junc_nodes:
            continue
        xs = [nodes[m].x for m in members]
        ys = [nodes[m].y for m in members]
        zs = [nodes[m].z for m in members]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        center_nid = min(members, key=lambda m: math.hypot(
            nodes[m].x - cx, nodes[m].y - cy, (nodes[m].z - cz) * 0.5))
        add_anchor("J%03d-C" % jid, "junction_center", center_nid, jid=jid)

        by_arm: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for src in enter_by_junc.get(jid, []):
            n = nodes[src]
            by_arm[(n.road_id, n.section_id)].append(src)
        # 每arm取离质心最近代表，再按方位合并同方位多arm（v2.1）
        arm_reps = []
        for (rid, sid), srcs in sorted(by_arm.items()):
            rep = min(srcs, key=lambda s: math.hypot(nodes[s].x - cx, nodes[s].y - cy))
            arm_reps.append(((rid, sid), rep))
        by_bearing: Dict[str, List[Tuple[Tuple[int, int], str]]] = defaultdict(list)
        for key, rep in arm_reps:
            # R2-P1-04：统一向量方位入口（+X=北在前的 atan2 罗盘角），
            # 替换旧 90.0-bear 换算（会造成 N↔E 镜像的 arm 归桶错误）
            compass = bearing_from_vector(nodes[rep].x - cx, nodes[rep].y - cy)
            by_bearing[compass].append((key, rep))
        for compass, reps in sorted(by_bearing.items()):
            rep = min((r for _k, r in reps),
                      key=lambda s: math.hypot(nodes[s].x - cx, nodes[s].y - cy))
            add_anchor("J%03d-E-%s" % (jid, compass), "junction_entry", rep, jid=jid)
            # 被合并掉的 arm 若 min_global 内无任何锚，补一个入口锚（防链端失锚）
            for _k, r in reps:
                if r == rep:
                    continue
                if all(math.hypot(nodes[r].x - a.x, nodes[r].y - a.y) >= min_global
                       for a in anchors):
                    add_anchor("J%03d-E-%s-r%03d" % (jid, compass, _k[0]),
                               "junction_entry_arm", r, jid=jid)
        if diag > big_junc_diag:
            bound = [(a.x, a.y) for a in anchors
                     if a.junction_id == jid and a.type != "junction_center"]
            if len(bound) >= 2:
                best = None
                for i in range(len(bound)):
                    for j in range(i + 1, len(bound)):
                        d = math.hypot(bound[i][0] - bound[j][0], bound[i][1] - bound[j][1])
                        if best is None or d > best[0]:
                            best = (d, bound[i], bound[j])
                mx = (best[1][0] + best[2][0]) / 2
                my = (best[1][1] + best[2][1]) / 2
                mid_nid = min(members, key=lambda m: math.hypot(nodes[m].x - mx, nodes[m].y - my))
                add_anchor("J%03d-M" % jid, "junction_mid", mid_nid, jid=jid)

    # ---------------- B. 路段锚（方向粒度链 + 夹逼修边 + 空档硬上限保底） ----------------
    external_ids = [nid for nid, n in nodes.items() if not n.is_junction]
    visited_edge: Set[Tuple[str, str]] = set()
    chains: List[List[str]] = []
    terminals = [nid for nid in external_ids
                 if follow_out.get(nid, 0) != 1 or follow_in.get(nid, 0) != 1]
    for t in terminals:
        for nxt in follow_adj.get(t, []):
            if (t, nxt) in visited_edge:
                continue
            visited_edge.add((t, nxt))
            chain = [t, nxt]
            cur = nxt
            while True:
                o = follow_adj.get(cur, [])
                if len(o) != 1 or follow_in.get(cur, 0) != 1:
                    break
                nn = o[0]
                if (cur, nn) in visited_edge:
                    break
                visited_edge.add((cur, nn))
                chain.append(nn)
                cur = nn
            if len(chain) >= 3:
                chains.append(chain)
    for e_src in list(follow_adj):
        for nxt in follow_adj[e_src]:
            if (e_src, nxt) in visited_edge:
                continue
            chain = [e_src, nxt]
            visited_edge.add((e_src, nxt))
            cur = nxt
            while True:
                o = follow_adj.get(cur, [])
                if len(o) != 1:
                    break
                nn = o[0]
                if nn == e_src or (cur, nn) in visited_edge:
                    break
                visited_edge.add((cur, nn))
                chain.append(nn)
                cur = nn
            if len(chain) >= 3:
                chains.append(chain)

    by_roadsec: Dict[Tuple[int, int], List[List[str]]] = defaultdict(list)
    for ch in chains:
        rids = {(nodes[n].road_id, nodes[n].section_id) for n in ch}
        by_roadsec[min(rids) if rids else (-1, -1)].append(ch)

    for (rid, sid), chs in sorted(by_roadsec.items()):
        pos = [c for c in chs if nodes[c[0]].lane_id > 0]
        neg = [c for c in chs if nodes[c[0]].lane_id < 0]
        sec_key = (rid, sid)
        for g in (pos, neg):
            if not g:
                continue
            # 同向多车道合并：取节点最多链，平手取 |lane_id| 小者（内侧），再取 lane_id 大者
            rep = sorted(g, key=lambda c: (-len(c), abs(nodes[c[0]].lane_id),
                                           -nodes[c[0]].lane_id))[0]
            ss = sorted(rep, key=lambda n: nodes[n].s_m)
            L = nodes[ss[-1]].s_m - nodes[ss[0]].s_m
            # 保底 n_floor=ceil(L/gap_max)-1：空档硬上限优先于 gap_min 下限
            n_floor = max(0, math.ceil(L / gap_max) - 1)
            n_mid = max(round(L / s_target) - 1, n_floor)
            while n_mid > 0 and L / (n_mid + 1) < gap_min and n_mid > n_floor:
                n_mid -= 1
            if n_mid <= 0:
                continue
            for k in range(1, n_mid + 1):
                target = nodes[ss[0]].s_m + (k / (n_mid + 1)) * L
                nid = min(ss, key=lambda n: abs(nodes[n].s_m - target))
                add_anchor("R%03d-%d-%s-%d" % (rid, sid, bearing_from_yaw(nodes[nid].yaw_deg), k),
                           "road_mid", nid, sec_key=sec_key)

    # ---------------- B2. 路口内部车道链锚 ----------------
    int_visited: Set[Tuple[str, str]] = set()
    int_chains: List[List[str]] = []
    for t, n in nodes.items():
        if not n.is_junction:
            continue
        if int_out.get(t, 0) < 1:
            continue
        if int_out.get(t, 0) == 1 and int_in.get(t, 0) == 1:
            continue
        for nxt in int_adj.get(t, []):
            if (t, nxt) in int_visited:
                continue
            int_visited.add((t, nxt))
            chain = [t, nxt]
            cur = nxt
            while int_out.get(cur, 0) == 1 and int_in.get(cur, 0) == 1:
                nn = int_adj[cur][0]
                if (cur, nn) in int_visited:
                    break
                int_visited.add((cur, nn))
                chain.append(nn)
                cur = nn
            if len(chain) >= 3:
                int_chains.append(chain)

    flow_count = 0
    for ch in int_chains:
        jids = {nodes[n].junction_id for n in ch}
        jid = next((j for j in jids if j is not None), None)
        if jid is None or jid not in junc_nodes or len(junc_nodes[jid]) < min_junc_nodes:
            continue
        cum = [0.0]
        for i in range(1, len(ch)):
            cum.append(cum[-1] + math.hypot(nodes[ch[i]].x - nodes[ch[i - 1]].x,
                                            nodes[ch[i]].y - nodes[ch[i - 1]].y))
        L = cum[-1]
        n_floor = max(0, math.ceil(L / gap_max) - 1)
        n_mid = max(round(L / s_target) - 1, n_floor)
        while n_mid > 0 and L / (n_mid + 1) < gap_min and n_mid > n_floor:
            n_mid -= 1
        if n_mid <= 0:
            continue
        for k in range(1, n_mid + 1):
            tgt = (k / (n_mid + 1)) * L
            i = min(range(len(ch)), key=lambda idx: abs(cum[idx] - tgt))
            add_anchor("J%03d-P%03d" % (jid, flow_count + k), "junction_path_mid",
                       ch[i], jid=jid)
        flow_count += n_mid

    # ---------------- C. 全局过滤（同路口/同路段豁免 + 同fine_node去重） ----------------
    anchors.sort(key=lambda a: (_PRIO.get(a.type, 3), a.name))
    kept: List[AnchorRecord] = []
    kept_nodes: Set[str] = set()
    for a in anchors:
        if a.fine_node in kept_nodes:
            continue
        ok = True
        for k in kept:
            if a.junction_id is not None and k.junction_id == a.junction_id:
                continue
            if a.section_key is not None and k.section_key == a.section_key:
                continue
            if math.hypot(a.x - k.x, a.y - k.y) < min_global:
                ok = False
                break
        if ok:
            kept.append(a)
            kept_nodes.add(a.fine_node)
    anchors = sorted(kept, key=lambda a: (a.type, a.name))

    # ---------------- D. fill 兜底（带重锚定的 multi-source Dijkstra） ----------------
    dist = {a.fine_node: 0.0 for a in anchors}
    pq = [(0.0, n) for n in dist]
    heapq.heapify(pq)
    fill_count = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, edge_list in store.successors(u).items():
            w = min(e.length_m for e in edge_list)
            nd = d + w
            if nd > gap_max:
                if v not in dist:
                    add_anchor("F%03d" % (fill_count + 1), "fill_gap", v)
                    fill_count += 1
                    dist[v] = 0.0
                    heapq.heappush(pq, (0.0, v))
            elif nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return anchors
