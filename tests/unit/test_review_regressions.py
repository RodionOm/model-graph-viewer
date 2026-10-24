from collections.abc import Hashable

import pytest

from model_graph_viewer import Graph, Node, apply_passes, load_graph
from model_graph_viewer.errors import ConfigurationError, GraphValidationError
from model_graph_viewer.renderers import GraphvizRenderer


@pytest.mark.parametrize(
    "field,value", [("name", None), ("name", 1), ("op_type", None), ("op_type", 5), ("op_type", "")]
)
def test_node_fields_rejected_at_construction(field, value):
    values = {"node_id": "0", "name": "x", "op_type": "Op", field: value}
    with pytest.raises(GraphValidationError):
        Node(**values)


def test_nodes_keep_value_equality_and_are_explicitly_unhashable():
    node = Node("0", "x", "Op")
    assert node == Node("0", "x", "Op")
    assert not isinstance(node, Hashable)
    node.attributes["axis"] = 1
    assert node != Node("0", "x", "Op")


def test_pass_may_return_same_graph():
    class ReturningPass:
        def apply(self, graph):
            graph.add_node(Node("0", "output", "Result"))
            return graph

    graph = Graph()
    assert apply_passes(graph, [ReturningPass()]) is graph
    assert list(graph.nodes) == ["0"]


def test_result_port_is_hidden_only_when_unconnected():
    from model_graph_viewer import Edge

    graph = Graph([Node("r", "result", "Result", output_ports=(0,))])
    renderer = GraphvizRenderer()
    assert 'PORT="out0"' not in renderer.to_dot(graph).source
    assert graph.nodes["r"].output_ports == (0,)
    graph.add_node(Node("c", "consumer", "Custom", input_ports=(0,)))
    graph.add_edge(Edge("r", 0, "c", 0))
    assert 'PORT="out0"' in renderer.to_dot(graph).source


def test_protocol_loader_needs_no_weights_argument():
    class Loader:
        def load(self, path):
            assert path == "m.custom"
            return Graph([Node("0", "custom", "Custom")])

    assert len(load_graph("m.custom", loader=Loader()).nodes) == 1
    with pytest.raises(ConfigurationError, match="omit weights_path"):
        load_graph("m.custom", loader=Loader(), weights_path="weights.bin")


@pytest.mark.parametrize("loader", [object(), type("BadLoader", (), {"load": lambda s, p: None})()])
def test_invalid_loader_contract(loader):
    with pytest.raises(ConfigurationError):
        load_graph("x", loader=loader)
