"""roadnet/store.py - 只读路网加载、版本校验 (v2)

关键改进：
- 支持多边（multigraph）：out_edges[src][dst] -> list[FineEdge]
- decision_edges 使用 decision_edge_id 索引
- 加载时校验 file_hashes
- fail-fast：不匹配时抛异常
"""
import os
import logging
from typing import Dict, List, Tuple
from .schema import FineNode, FineEdge, DecisionNode, DecisionEdge, GraphManifest
from .serializer import load_graph

logger = logging.getLogger("LiteV1.RoadnetStore")


class RoadGraphStore:
    """只读冻结路网存储"""

    def __init__(self, graph_dir: str, nodes: Dict[str, FineNode], edges: Dict[str, FineEdge],
                 decision_nodes: Dict[str, DecisionNode], decision_edges: Dict[str, DecisionEdge],
                 manifest: GraphManifest):
        self.graph_dir = graph_dir
        self.nodes = nodes
        self.edges = edges
        self.decision_nodes = decision_nodes
        self.decision_edges = decision_edges
        self.manifest = manifest

        # Build adjacency lists - multigraph: src -> dst -> list of edges
        self.out_edges: Dict[str, Dict[str, List[FineEdge]]] = {}
        self.in_edges: Dict[str, Dict[str, List[FineEdge]]] = {}
        for e in edges.values():
            self.out_edges.setdefault(e.src, {}).setdefault(e.dst, []).append(e)
            self.in_edges.setdefault(e.dst, {}).setdefault(e.src, []).append(e)

        # Decision adjacency - multigraph
        self.out_decision_edges: Dict[str, Dict[str, List[DecisionEdge]]] = {}
        self.in_decision_edges: Dict[str, Dict[str, List[DecisionEdge]]] = {}
        for e in decision_edges.values():
            self.out_decision_edges.setdefault(e.src, {}).setdefault(e.dst, []).append(e)
            self.in_decision_edges.setdefault(e.dst, {}).setdefault(e.src, []).append(e)

        # Fine node spatial index (road/section/lane -> sorted by s)
        from collections import defaultdict
        self._lane_index: Dict[Tuple[int, int, int], List[FineNode]] = defaultdict(list)
        for node in nodes.values():
            key = (node.road_id, node.section_id, node.lane_id)
            self._lane_index[key].append(node)
        for key in self._lane_index:
            self._lane_index[key].sort(key=lambda n: n.s_m)

    @classmethod
    def load(cls, graph_dir: str, expected_map_key: str = None,
             expected_graph_id: str = None,
             expected_opendrive_sha256: str = None,
             verify_hashes: bool = True,
             require_frozen: bool = True) -> "RoadGraphStore":
        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"Road graph directory not found: {graph_dir}")
        nodes, edges, d_nodes, d_edges, manifest = load_graph(graph_dir, verify_hashes=verify_hashes)

        if expected_map_key and manifest.map_key != expected_map_key:
            raise ValueError(f"Map key mismatch: expected {expected_map_key}, got {manifest.map_key}")
        if expected_graph_id and manifest.graph_id != expected_graph_id:
            raise ValueError(f"Graph ID mismatch: expected {expected_graph_id}, got {manifest.graph_id}")
        if expected_opendrive_sha256 and manifest.opendrive_sha256 != expected_opendrive_sha256:
            raise ValueError(f"OpenDRIVE SHA-256 mismatch for graph {manifest.graph_id}")
        if require_frozen:
            if manifest.status != "frozen":
                raise ValueError(f"Graph status is not frozen: {manifest.status} (graph: {manifest.graph_id})")
            # Verify FROZEN marker
            frozen_path = os.path.join(graph_dir, "FROZEN")
            if not os.path.exists(frozen_path):
                raise ValueError(f"FROZEN marker not found in {graph_dir}")

        return cls(graph_dir, nodes, edges, d_nodes, d_edges, manifest)

    def get_node(self, node_id: str) -> FineNode:
        return self.nodes[node_id]

    def get_decision_node(self, did: str) -> DecisionNode:
        return self.decision_nodes[did]

    def successors(self, node_id: str) -> Dict[str, List[FineEdge]]:
        """返回 dst -> list of edges (multigraph)"""
        return self.out_edges.get(node_id, {})

    def predecessors(self, node_id: str) -> Dict[str, List[FineEdge]]:
        return self.in_edges.get(node_id, {})

    def decision_successors(self, did: str) -> Dict[str, List[DecisionEdge]]:
        return self.out_decision_edges.get(did, {})

    def decision_predecessors(self, did: str) -> Dict[str, List[DecisionEdge]]:
        return self.in_decision_edges.get(did, {})

    def get_lane_nodes(self, road_id: int, section_id: int, lane_id: int) -> List[FineNode]:
        return self._lane_index.get((road_id, section_id, lane_id), [])
