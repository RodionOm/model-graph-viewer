"""The minimal contract for a visualization transformation."""

from typing import Protocol

from model_graph_viewer.graph import Graph


class TransformationPass(Protocol):
    def apply(self, graph: Graph) -> Graph | None:
        """Modify in place; return None or the same graph, leaving it valid."""
        ...
