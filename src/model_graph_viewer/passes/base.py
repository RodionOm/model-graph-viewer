"""The minimal contract for a visualization transformation."""

from typing import Protocol

from model_graph_viewer.graph import Graph


class TransformationPass(Protocol):
    def apply(self, graph: Graph) -> None:
        """Modify the supplied graph in place and leave it structurally valid."""
        ...
