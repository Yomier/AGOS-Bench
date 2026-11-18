"""roadnet/serializer.py - 路网序列化与校验 (v2)

关键改进（按外部 Agent 审计）：
- 所有记录按稳定 ID 排序（P0-3.10）
- gzip 使用 mtime=0 保证确定性
- canonical asset hash 基于未压缩排序后 JSON bytes
- RoadGraphStore.load() 校验全部文件 hash
- 用 staging 目录原子提交
"""
import json
import hashlib
import gzip
import os
import io
import logging
from typing import Dict, Tuple, List

from .schema import FineNode, FineEdge, DecisionNode, DecisionEdge, GraphManifest

logger = logging.getLogger("LiteV1.Serializer")


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_jsonl_gz_deterministic(items: List[dict], path: str):
    """确定性序列化：排序 + gzip mtime=0"""
    sorted_items = sorted(items, key=lambda d: _sort_key(d))
    buf = io.StringIO()
    for item in sorted_items:
        buf.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    content = buf.getvalue().encode("utf-8")
    with gzip.GzipFile(path, "wb", compresslevel=6, mtime=0) as f:
        f.write(content)


def _sort_key(d: dict) -> str:
    """稳定排序键"""
    for k in ("node_id", "edge_id", "decision_node_id", "decision_edge_id"):
        if k in d:
            return str(d[k])
    return json.dumps(d, sort_keys=True)


def _load_jsonl_gz(path: str) -> list:
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def serialize_graph(graph_dir: str, nodes: Dict[str, FineNode], edges: Dict[str, FineEdge],
                    decision_nodes: Dict[str, DecisionNode], decision_edges: Dict[str, DecisionEdge],
                    manifest: GraphManifest) -> str:
    """序列化图资产到目录，返回 canonical asset SHA-256"""
    os.makedirs(graph_dir, exist_ok=True)

    fine_nodes_path = os.path.join(graph_dir, "fine_nodes.jsonl.gz")
    fine_edges_path = os.path.join(graph_dir, "fine_edges.jsonl.gz")
    decision_nodes_path = os.path.join(graph_dir, "decision_nodes.jsonl.gz")
    decision_edges_path = os.path.join(graph_dir, "decision_edges.jsonl.gz")
    manifest_path = os.path.join(graph_dir, "manifest.json")

    _save_jsonl_gz_deterministic([n.to_dict() for n in nodes.values()], fine_nodes_path)
    _save_jsonl_gz_deterministic([e.to_dict() for e in edges.values()], fine_edges_path)
    _save_jsonl_gz_deterministic([n.to_dict() for n in decision_nodes.values()], decision_nodes_path)
    _save_jsonl_gz_deterministic([e.to_dict() for e in decision_edges.values()], decision_edges_path)

    # Compute file hashes
    file_hashes = {
        "fine_nodes": _sha256_of_file(fine_nodes_path),
        "fine_edges": _sha256_of_file(fine_edges_path),
        "decision_nodes": _sha256_of_file(decision_nodes_path),
        "decision_edges": _sha256_of_file(decision_edges_path),
    }

    # Canonical asset hash: over uncompressed sorted JSON bytes
    canonical_parts = []
    for key in sorted(file_hashes.keys()):
        canonical_parts.append(f"{key}:{file_hashes[key]}")
    canonical_content = "|".join(canonical_parts).encode("utf-8")
    asset_hash = _sha256_of_bytes(canonical_content)

    manifest_data = manifest.to_dict()
    manifest_data["file_hashes"] = file_hashes
    manifest_data["asset_sha256"] = asset_hash

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False, sort_keys=True)

    return asset_hash


def load_graph(graph_dir: str, verify_hashes: bool = True) -> Tuple[
        Dict[str, FineNode], Dict[str, FineEdge],
        Dict[str, DecisionNode], Dict[str, DecisionEdge],
        GraphManifest]:
    """从目录加载图资产，可选校验文件 hash"""
    manifest_path = os.path.join(graph_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest = GraphManifest.from_dict(manifest_data)

    fine_nodes_path = os.path.join(graph_dir, "fine_nodes.jsonl.gz")
    fine_edges_path = os.path.join(graph_dir, "fine_edges.jsonl.gz")
    decision_nodes_path = os.path.join(graph_dir, "decision_nodes.jsonl.gz")
    decision_edges_path = os.path.join(graph_dir, "decision_edges.jsonl.gz")

    if verify_hashes and manifest.file_hashes:
        for name, path in [("fine_nodes", fine_nodes_path), ("fine_edges", fine_edges_path),
                           ("decision_nodes", decision_nodes_path), ("decision_edges", decision_edges_path)]:
            if name not in manifest.file_hashes:
                continue
            actual = _sha256_of_file(path)
            expected = manifest.file_hashes[name]
            if actual != expected:
                raise ValueError(f"File hash mismatch for {name}: expected {expected[:16]}, got {actual[:16]}")

    nodes = {d["node_id"]: FineNode.from_dict(d) for d in _load_jsonl_gz(fine_nodes_path)}
    edges = {d["edge_id"]: FineEdge.from_dict(d) for d in _load_jsonl_gz(fine_edges_path)}
    d_nodes = {d["decision_node_id"]: DecisionNode.from_dict(d) for d in _load_jsonl_gz(decision_nodes_path)}
    d_edges = {}
    for d in _load_jsonl_gz(decision_edges_path):
        e = DecisionEdge.from_dict(d)
        d_edges[e.decision_edge_id] = e

    return nodes, edges, d_nodes, d_edges, manifest


def atomic_freeze(staging_dir: str, final_dir: str, manifest: GraphManifest):
    """原子提交：验证 staging -> 写 FROZEN -> rename 到 final"""
    frozen_path = os.path.join(staging_dir, "FROZEN")
    with open(frozen_path, "w", encoding="utf-8") as f:
        f.write(f"{manifest.graph_id}\n{manifest.asset_sha256}\n")
    # Atomic rename (on same filesystem)
    if os.path.exists(final_dir):
        import shutil
        shutil.rmtree(final_dir)
    os.rename(staging_dir, final_dir)
    logger.info(f"Atomic freeze: {staging_dir} -> {final_dir}")
