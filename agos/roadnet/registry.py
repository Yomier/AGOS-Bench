"""roadnet/registry.py - 通过注册表解析冻结资产 (P0-3.2)

外部 Agent 要求：配置指向注册表 JSON，loader 通过 map_key 解析到具体图目录。
"""
import json
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("LiteV1.RoadGraphRegistry")


class RoadGraphRegistry:
    """路网注册表：从 map_key 解析到具体冻结图目录"""

    def __init__(self, registry_path: str, data: Dict[str, Any]):
        self.registry_path = registry_path
        self.data = data

    @classmethod
    def load(cls, registry_path: str) -> "RoadGraphRegistry":
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Road graph registry not found: {registry_path}")
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(registry_path, data)

    def resolve(self, map_key: str,
                expected_graph_id: Optional[str] = None,
                expected_asset_sha256: Optional[str] = None) -> str:
        """通过 map_key 解析冻结图目录路径

        Args:
            map_key: 逻辑地图名 (如 "Town03")
            expected_graph_id: 可选校验
            expected_asset_sha256: 可选校验

        Returns:
            图目录绝对路径

        Raises:
            KeyError: map_key 不在注册表中
            ValueError: 校验失败
        """
        entries = self.data.get("entries", {})
        if map_key not in entries:
            raise KeyError(f"Map key '{map_key}' not found in registry")
        entry = entries[map_key]
        graph_dir = entry["graph_dir"]
        # P1-09 (2026-09-04 r2 external review): 相对 graph_dir 相对注册表文件
        # 所在目录解析，而不是进程 CWD——注册表随资产目录整体迁移/跨主机部署时
        # 无需改写绝对路径。
        if not os.path.isabs(graph_dir):
            graph_dir = os.path.join(os.path.dirname(os.path.abspath(self.registry_path)),
                                     graph_dir)
        graph_dir = os.path.normpath(graph_dir)
        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
        if expected_graph_id and entry.get("graph_id") != expected_graph_id:
            raise ValueError(f"Graph ID mismatch for {map_key}: expected {expected_graph_id}, "
                           f"got {entry.get('graph_id')}")
        if expected_asset_sha256 and entry.get("asset_sha256") != expected_asset_sha256:
            raise ValueError(f"Asset SHA-256 mismatch for {map_key}: expected {expected_asset_sha256}, "
                           f"got {entry.get('asset_sha256')}")
        return graph_dir

    def get_entry(self, map_key: str) -> Dict[str, Any]:
        entries = self.data.get("entries", {})
        if map_key not in entries:
            raise KeyError(f"Map key '{map_key}' not found in registry")
        return entries[map_key]

    def list_maps(self) -> list:
        return list(self.data.get("entries", {}).keys())


def save_registry(output_dir: str, summary: Dict[str, Any], registry_data: Dict[str, Any] = None):
    """生成/更新永久定式注册表"""
    if registry_data is None:
        import datetime
        registry_data = {
            "schema_version": 2,
            "generated_at": datetime.datetime.now().isoformat(),
            "fine_resolution_m": 2.0,
            "decision_spacing_m": 10.0,
            "lane_change_policy": "none",
            "entries": {}
        }
    for map_name, info in summary.items():
        if info.get("status") != "ok":
            continue
        graph_dir = info.get("graph_dir") or ""
        # P1-09: 图目录位于 output_dir 下时存相对路径，注册表可随资产目录整体迁移
        if graph_dir.startswith(os.path.abspath(output_dir) + os.sep):
            graph_dir = os.path.relpath(graph_dir, os.path.abspath(output_dir))
        registry_data["entries"][map_name] = {
            "graph_id": info.get("graph_id"),
            "graph_dir": graph_dir,
            "asset_sha256": info.get("asset_hash"),
            "opendrive_sha256": info.get("opendrive_sha256"),
            "carla_asset_name": info.get("carla_asset_name"),
            "map_key": map_name,
        }
    path = os.path.join(output_dir, "road_graph_registry.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info(f"Road graph registry saved to {path}")
