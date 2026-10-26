"""Replace high-fan-out drawing with references, retaining every graph edge."""

from model_graph_viewer.graph import Graph


class FanoutReferencesPass:
    """Mark sources with strictly more than threshold outgoing edges.

    Count connections across all output ports, including two connections to one
    consumer. The renderer labels both ends instead of drawing those edges.
    Run after structural passes such as HideConstantsPass to count visible edges.
    Reapplying recomputes the selection; graph connectivity is never changed.
    """

    def __init__(self, threshold: int = 3) -> None:
        if type(threshold) is not int or threshold < 1:
            raise ValueError("fan-out threshold must be a positive integer.")
        self.threshold = threshold

    def apply(self, graph: Graph) -> None:
        for node in graph.nodes.values():
            if len(graph.outgoing(node.node_id)) > self.threshold:
                node.visualization["reference_fanout"] = True
            else:
                node.visualization.pop("reference_fanout", None)
