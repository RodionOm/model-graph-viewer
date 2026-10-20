import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from model_graph_viewer import Dimension, Edge, Graph, Node, TensorMetadata, TensorShape
from model_graph_viewer.errors import GraphValidationError


def branching_graph():
    return Graph(
        [
            Node("s", "duplicate", "Split", input_ports=(0,), output_ports=(0, 1)),
            Node("a", "duplicate", "CustomOperation", input_ports=(0, 1, 7), output_ports=(0,)),
            Node("b", "output", "Result", input_ports=(0,)),
        ],
        [
            Edge("s", 0, "a", 0, TensorMetadata(TensorShape((None, 768)), "fp32")),
            Edge("s", 1, "a", 7),
            Edge("s", 0, "b", 0),
        ],
    )


def test_duplicate_names_and_parallel_connections_keep_port_identity():
    graph = branching_graph()
    graph.validate()
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 3
    assert [(e.from_port, e.to_port) for e in graph.incoming("a")] == [(0, 0), (1, 7)]
    assert graph.nodes["a"].name == graph.nodes["s"].name
    assert graph.nodes["a"].op_type == "CustomOperation"
    assert len(graph.outgoing("s")) == 3
    assert graph.edges[("a", 0)].tensor.shape == TensorShape((None, 768))


def test_one_output_can_connect_to_two_inputs_of_same_node():
    graph = branching_graph()
    graph.add_edge(Edge("s", 0, "a", 1))
    graph.validate()
    assert len(graph.incoming("a")) == 3
    assert len(graph.outgoing("s")) == 4


def test_edge_removal_preserves_ports_and_unrelated_connections():
    graph = branching_graph()
    snapshot = graph.incoming("a")
    removed = graph.remove_edge("a", 0)
    graph.validate()
    assert removed == snapshot[0]
    assert len(snapshot) == 2
    assert graph.nodes["a"].input_ports == (0, 1, 7)
    assert graph.incoming("a") == (Edge("s", 1, "a", 7),)
    assert graph.incoming("b") == (Edge("s", 0, "b", 0),)


def test_remove_node_cleans_both_indexes_and_preserves_other_nodes():
    graph = branching_graph()
    graph.add_edge(Edge("a", 0, "s", 0))
    graph.remove_node("a")
    graph.validate()
    assert set(graph.nodes) == {"s", "b"}
    assert tuple(graph.edges.values()) == (Edge("s", 0, "b", 0),)
    assert graph.incoming("s") == ()
    assert graph.outgoing("s") == (Edge("s", 0, "b", 0),)


def test_remove_node_handles_self_loop_once():
    graph = Graph([Node("n", "loop", "Custom", (0,), (0,))], [Edge("n", 0, "n", 0)])
    graph.validate()
    graph.remove_node("n")
    graph.validate()
    assert not graph.nodes
    assert not graph.edges


@pytest.mark.parametrize(
    ("edge", "message"),
    [
        (Edge("missing", 0, "a", 1), "does not exist"),
        (Edge("s", 0, "missing", 0), "does not exist"),
        (Edge("s", 2, "a", 1), "no output port"),
        (Edge("s", 0, "a", 2), "no input port"),
        (Edge("s", 1, "a", 0), "already has a source"),
        (Edge("s", 0, "a", 0), "already has a source"),
    ],
)
def test_failed_edge_insertion_is_atomic(edge, message):
    graph = branching_graph()
    before = dict(graph.edges)
    with pytest.raises(GraphValidationError, match=message):
        graph.add_edge(edge)
    graph.validate()
    assert dict(graph.edges) == before


def test_duplicate_id_rejected_without_replacing_original_node():
    graph = branching_graph()
    original = graph.nodes["a"]
    with pytest.raises(GraphValidationError, match="Duplicate node ID"):
        graph.add_node(Node("a", "other name", "OtherType"))
    assert graph.nodes["a"] is original
    graph.validate()


@pytest.mark.parametrize(
    "method,args",
    [
        ("incoming", ("missing",)),
        ("outgoing", ("missing",)),
        ("remove_node", ("missing",)),
        ("remove_edge", ("a", 99)),
    ],
)
def test_invalid_lookup_or_removal_preserves_graph(method, args):
    graph = branching_graph()
    with pytest.raises(GraphValidationError):
        getattr(graph, method)(*args)
    graph.validate()
    assert len(graph.edges) == 3


@pytest.mark.parametrize("ports", [(-1,), (0, 0), (True,), (1.5,)])
def test_invalid_node_ports_are_rejected(ports):
    with pytest.raises(GraphValidationError):
        Node("a", "a", "Add", input_ports=ports)


@pytest.mark.parametrize("port", [-1, True, 0.5])
def test_invalid_edge_ports_are_rejected(port):
    with pytest.raises(GraphValidationError):
        Edge("a", port, "b", 0)


def test_structure_is_read_only_but_visual_metadata_can_change():
    graph = branching_graph()
    with pytest.raises(TypeError):
        graph.nodes["new"] = Node("new", "new", "Add")
    with pytest.raises(TypeError):
        del graph.edges[("a", 0)]
    with pytest.raises(FrozenInstanceError):
        graph.nodes["a"].node_id = "different"
    graph.nodes["a"].visualization["border_color"] = "red"
    assert graph.nodes["a"].attributes == {}
    assert graph.nodes["s"].visualization == {}
    graph.validate()


def test_attribute_dictionaries_are_owned_by_each_node():
    attributes = {"axis": 1}
    node = Node("a", "a", "Split", attributes=attributes)
    attributes["axis"] = 2
    assert node.attributes == {"axis": 1}


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (None, "[unknown]"),
        (TensorShape(None), "[...]"),
        (TensorShape(()), "[]"),
        (TensorShape((0, 768)), "[0, 768]"),
        (TensorShape((None, 768)), "[?, 768]"),
        (TensorShape((Dimension(1, 8), Dimension(4))), "[1..8, 4..?]"),
        (TensorShape((Dimension(), Dimension(5, 5))), "[?, 5]"),
    ],
)
def test_shape_states_remain_distinct(shape, expected):
    assert str(TensorMetadata(shape, "fp32")) == f"fp32{expected}"
    assert str(TensorMetadata(shape)) == f"?{expected}"


@pytest.mark.parametrize("value", [-1, True, "?", 3.5])
def test_invalid_dimension_rejected(value):
    with pytest.raises(GraphValidationError):
        TensorShape((value,))


@pytest.mark.parametrize("lower,upper", [(-1, None), (0, -1), (5, 2)])
def test_invalid_dimension_interval_rejected(lower, upper):
    with pytest.raises(GraphValidationError):
        Dimension(lower, upper)


def test_validate_detects_broken_adjacency_index():
    graph = branching_graph()
    graph._outgoing["s"].pop(("a", 7))
    with pytest.raises(GraphValidationError, match="Outgoing edge index"):
        graph.validate()


def test_empty_graph_is_valid():
    Graph().validate()


def test_graph_api_does_not_import_model_or_rendering_backends():
    code = (
        "import sys; from model_graph_viewer import Graph; Graph().validate(); "
        "assert 'openvino' not in sys.modules; assert 'graphviz' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
