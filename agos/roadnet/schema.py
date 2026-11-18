"""roadnet/schema.py - 冻结路网数据结构 (schema v2)

schema v2 变更（按外部 Agent 审计报告）：
- GraphManifest 增加 map_key / carla_asset_name / source_bundle_sha256 / build_config_sha256 / validation_status
- FineEdge 细分 junction edge 类型: JUNCTION_ENTER / JUNCTION_INTERNAL / JUNCTION_EXIT
- FineEdge 增加 geometry 字段（lane-change macro 采样点）
- DecisionNode 改为 roles 列表，独立 ID D:<anchor_fine_node_id>
- DecisionEdge 增加 decision_edge_id，支持 multigraph
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, Dict, Any, List


@dataclass(frozen=True)
class FineNode:
    """2m 有向执行图节点"""
    node_id: str
    carla_waypoint_id: int
    road_id: int
    section_id: int
    lane_id: int
    s_m: float
    x: float
    y: float
    z: float
    yaw_deg: float
    lane_width_m: float
    is_junction: bool
    junction_id: Optional[int]
    lane_type: str  # Driving / Bidirectional

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FineNode":
        return cls(**d)


@dataclass(frozen=True)
class FineEdge:
    """2m 有向执行图边

    edge_type:
        LANE_FOLLOW       - 普通车道跟随 (src/dst 均非 junction)
        JUNCTION_ENTER    - 进入路口 (src 非 junction, dst junction)
        JUNCTION_INTERNAL - 路口内部 (src/dst 均 junction)
        JUNCTION_EXIT     - 离开路口 (src junction, dst 非 junction)
        LANE_CHANGE_LEFT  - 合法左换道 macro
        LANE_CHANGE_RIGHT - 合法右换道 macro
        MAP_NATIVE_UTURN  - CARLA 原生掉头
    """
    edge_id: str
    src: str
    dst: str
    edge_type: str
    length_m: float
    base_travel_time_s: float
    turn_hint: Optional[str]  # LEFT, RIGHT, STRAIGHT, UTURN, None
    junction_id: Optional[int]
    # lane-change macro 的几何采样点: [(x, y, z, yaw), ...]
    # 普通 longitudinal edge 为空 tuple
    geometry: Tuple[Tuple[float, float, float, float], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["geometry"] = [list(g) for g in self.geometry]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FineEdge":
        d = dict(d)
        geo = d.get("geometry", [])
        if geo and isinstance(geo[0], (list, tuple)):
            d["geometry"] = tuple(tuple(g) for g in geo)
        else:
            d["geometry"] = tuple()
        return cls(**d)

    @property
    def is_longitudinal(self) -> bool:
        """是否为纵向行驶边（不含 lane-change）"""
        return self.edge_type in (
            "LANE_FOLLOW", "JUNCTION_ENTER", "JUNCTION_INTERNAL",
            "JUNCTION_EXIT", "MAP_NATIVE_UTURN"
        )


@dataclass(frozen=True)
class DecisionNode:
    """10m 决策图节点

    roles 可包含: CRUISE_ANCHOR, JUNCTION_ENTRY, JUNCTION_EXIT, DEAD_END, SPAWN_ANCHOR, FORK
    一个节点可同时是 JUNCTION_ENTRY 和 DEAD_END
    """
    decision_node_id: str  # D:<anchor_fine_node_id>
    anchor_fine_node_id: str
    roles: Tuple[str, ...]
    x: float
    y: float
    z: float
    yaw_deg: float
    junction_id: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["roles"] = list(self.roles)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionNode":
        d = dict(d)
        d["roles"] = tuple(d.get("roles", []))
        return cls(**d)

    @property
    def node_type(self) -> str:
        """兼容旧代码：返回第一个 role"""
        return self.roles[0] if self.roles else "CRUISE_ANCHOR"


@dataclass(frozen=True)
class DecisionEdge:
    """决策图宏边，保存完整 2m 路径

    edge_kind:
        CRUISE       - 道路走廊压缩边
        JUNCTION     - 路口 movement 边 (entry -> exit)
        LANE_CHANGE  - 换道 macro (暂不用)
    """
    decision_edge_id: str  # 唯一 ID: src->dst#index
    src: str  # decision_node_id
    dst: str  # decision_node_id
    edge_kind: str  # CRUISE / JUNCTION / LANE_CHANGE
    fine_path: Tuple[str, ...]
    path_length_m: float
    estimated_time_s: float
    maneuver: str  # FOLLOW, LEFT, RIGHT, STRAIGHT, UTURN, BACKTRACK
    junction_id: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fine_path"] = list(self.fine_path)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionEdge":
        d = dict(d)
        d["fine_path"] = tuple(d.get("fine_path", []))
        return cls(**d)


@dataclass(frozen=True)
class GraphManifest:
    """路网资产版本化清单 (schema v2)"""
    schema_version: int  # 2
    graph_id: str
    status: str  # frozen / provisional
    map_key: str  # 逻辑地图名: Town03
    carla_asset_name: str  # CARLA 资产路径: /Game/Carla/Maps/Town03_Opt
    carla_map_reported_name: str  # client.get_world().get_map().name
    opendrive_sha256: str
    source_bundle_sha256: str
    build_config_sha256: str
    carla_client_version: str
    carla_server_version: str
    carla_air_version: str
    fine_resolution_m: float
    decision_spacing_m: float
    junction_lookahead_m: float
    included_lane_types: Tuple[str, ...]
    lane_change_policy: str  # none / legal_macro_v1
    normal_edges_use_previous: bool
    allow_straight_line_fallback: bool
    fine_node_count: int
    fine_edge_count: int
    decision_node_count: int
    decision_edge_count: int
    asset_sha256: str
    generator_commit: str
    build_timestamp: str
    validation_status: str  # passed / failed / not_run
    target_maps: Tuple[str, ...] = ()
    file_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["included_lane_types"] = list(self.included_lane_types)
        d["target_maps"] = list(self.target_maps)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphManifest":
        d = dict(d)
        d["included_lane_types"] = tuple(d.get("included_lane_types", ["Driving"]))
        d["target_maps"] = tuple(d.get("target_maps", []))
        # 兼容 schema v1 字段
        d.setdefault("map_key", d.get("map_name", ""))
        d.setdefault("carla_asset_name", "")
        d.setdefault("carla_map_reported_name", "")
        d.setdefault("source_bundle_sha256", "")
        d.setdefault("build_config_sha256", "")
        d.setdefault("carla_client_version", "")
        d.setdefault("carla_server_version", d.get("carla_version", ""))
        d.setdefault("carla_air_version", "0.1.7")
        d.setdefault("build_timestamp", "")
        d.setdefault("validation_status", "not_run")
        return cls(**d)
