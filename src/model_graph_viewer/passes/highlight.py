"""Highlight all nodes whose names exactly match the supplied selection."""

from collections.abc import Iterable

from model_graph_viewer.graph import Graph


class HighlightPass:
    def __init__(self, node_names: Iterable[str]) -> None:
        if isinstance(node_names, str):
            raise TypeError("node_names must be an iterable of names, not a single string.")
        names = tuple(node_names)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("Each node name must be a string.")
        self.node_names = frozenset(names)

    def apply(self, graph: Graph) -> None:
        """Set only matching nodes' border color; preserve all other metadata."""
        for node in graph.nodes.values():
            if node.name in self.node_names:
                node.visualization["border_color"] = "red"
