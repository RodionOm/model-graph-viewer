"""Minimal extension examples; run from the repository root or install as a module."""

import json
from pathlib import Path

from model_graph_viewer import Edge, Graph, Node


class JsonGraphLoader:
    """Demonstration format: ordinary nodes/edges, without tensor metadata."""

    def load(self, model_path: str | Path) -> Graph:
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        return Graph(
            (Node(**node) for node in data["nodes"]),
            (Edge(**edge) for edge in data.get("edges", [])),
        )


class MarkOutputs:
    def __init__(self, color: str = "blue") -> None:
        self.color = color

    def apply(self, graph: Graph) -> Graph:
        for node in graph.nodes.values():
            if node.op_type == "Result":
                node.visualization["border_color"] = self.color
        return graph
