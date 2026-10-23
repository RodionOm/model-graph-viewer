"""Replace constant edges with port-specific visualization metadata."""

from model_graph_viewer.errors import GraphValidationError
from model_graph_viewer.graph import ConstantInput, Graph


class HideConstantsPass:
    """Hide Constant outputs, marking constants used by distinct consumer nodes.

    Existing operation attributes are untouched. The reserved visualization key
    'hidden_constants' maps destination input ports to ConstantInput records.
    """

    def apply(self, graph: Graph) -> None:
        # Snapshot connections before edits so shared markers do not depend on
        # removal order. Adjacency lookup visits each constant edge only once.
        constants = {
            node.node_id: graph.outgoing(node.node_id)
            for node in graph.nodes.values()
            if node.op_type == "Constant"
        }
        updates: dict[str, dict[int, ConstantInput]] = {}
        for constant_id, edges in constants.items():
            shared = len({edge.to_node for edge in edges}) > 1
            for edge in edges:
                if edge.to_node not in updates:
                    existing = graph.nodes[edge.to_node].visualization.get("hidden_constants", {})
                    if not isinstance(existing, dict) or any(
                        type(port) is not int
                        or not isinstance(record, ConstantInput)
                        or record.input_port != port
                        for port, record in existing.items()
                    ):
                        raise GraphValidationError(
                            f"Node {edge.to_node!r} has incompatible hidden_constants metadata."
                        )
                    # Copy only this small metadata mapping, never the graph or weights.
                    updates[edge.to_node] = dict(existing)
                annotations = updates[edge.to_node]
                if edge.to_port in annotations:
                    raise GraphValidationError(
                        f"Node {edge.to_node!r} already has hidden constant metadata "
                        f"for input port {edge.to_port}."
                    )
                annotations[edge.to_port] = ConstantInput(
                    input_port=edge.to_port,
                    constant_id=constant_id,
                    output_port=edge.from_port,
                    tensor=edge.tensor,
                    shared=shared,
                )

        # All metadata conflicts are checked before modifying the graph.
        for node_id, annotations in updates.items():
            graph.nodes[node_id].visualization["hidden_constants"] = annotations
        for edges in constants.values():
            for edge in edges:
                graph.remove_edge(edge.to_node, edge.to_port)
        for node_id in constants:
            if not graph.incoming(node_id) and not graph.outgoing(node_id):
                graph.remove_node(node_id)
