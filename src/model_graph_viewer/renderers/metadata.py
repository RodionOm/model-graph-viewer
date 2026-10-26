"""JSON inspection data; never read weights or stringify opaque backend objects."""

import math
from dataclasses import asdict

from model_graph_viewer.graph import ConstantInput, Graph


def _value(value: object, ancestors: frozenset[int] = frozenset()) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, ConstantInput):
        return _value(asdict(value), ancestors)
    if id(value) in ancestors:
        return {"unavailable": "cyclic attribute"}
    ancestors = ancestors | {id(value)}
    if isinstance(value, (list, tuple)):
        return [_value(item, ancestors) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _value(item, ancestors) for key, item in value.items()}
    if isinstance(value, dict):
        # Preserve non-string keys without collisions from str(key) conversion.
        return {
            "entries": [
                [_value(key, ancestors), _value(item, ancestors)] for key, item in value.items()
            ]
        }
    return {"unavailable_type": type(value).__name__}


def graph_metadata(graph: Graph, *, compact: bool) -> dict:
    """Describe the graph after passes, including every undrawn connection.

    Shapes retain missing metadata, dynamic rank, scalar and interval distinctions.
    Hidden constants are recorded on consumers; removed nodes are not resurrected.
    """
    graph.validate()
    return {
        "schema_version": 1,
        "compact_labels": compact,
        "connectivity": "All edges after structural passes; referenced edges remain present. "
        "Hidden Constant connections are stored under each node's hidden_constants.",
        "nodes": {
            node.node_id: {
                "svg_id": f"node_n{index}",
                "name": node.name,
                "op_type": node.op_type,
                "input_ports": list(node.input_ports),
                "output_ports": list(node.output_ports),
                "attributes": _value(
                    {
                        key: value
                        for key, value in node.attributes.items()
                        if node.op_type != "Constant" or key in {"dtype", "shape", "element_type"}
                    }
                ),
                "hidden_constants": [
                    asdict(record)
                    for record in node.visualization.get("hidden_constants", {}).values()
                ],
                "visualization": _value(
                    {
                        key: value
                        for key, value in node.visualization.items()
                        if key != "hidden_constants"
                    }
                ),
            }
            for index, node in enumerate(graph.nodes.values())
        },
        "edges": [
            {
                **asdict(edge),
                "rendered": not graph.nodes[edge.from_node].visualization.get(
                    "reference_fanout", False
                ),
            }
            for edge in graph.edges.values()
        ],
    }
