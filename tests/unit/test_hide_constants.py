from copy import deepcopy

import pytest

from model_graph_viewer import (
    ConstantInput,
    Dimension,
    Edge,
    Graph,
    Node,
    TensorMetadata,
    TensorShape,
)
from model_graph_viewer.errors import GraphValidationError
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, PassStep, Pipeline


@pytest.fixture
def graph():
    return Graph(
        [
            Node("x", "input", "Parameter", output_ports=(0,)),
            Node("w", "weights", "Constant", output_ports=(0,)),
            Node(
                "a",
                "matmul",
                "MatMul",
                input_ports=(0, 7),
                output_ports=(0,),
                attributes={"[7]": "original", "hidden_constants": "operation attribute"},
                visualization={"border_color": "red", "note": "keep"},
            ),
            Node("r", "output", "Result", input_ports=(0,)),
        ],
        [
            Edge("x", 0, "a", 0),
            Edge("w", 0, "a", 7, TensorMetadata(TensorShape((1024, 768)), "fp32")),
            Edge("a", 0, "r", 0),
        ],
    )


def test_single_constant_preserves_ports_attributes_and_unrelated_edges(graph):
    original_nodes = deepcopy(dict(graph.nodes))
    visible_edges = {key: edge for key, edge in graph.edges.items() if edge.from_node != "w"}
    metadata = graph.edges[("a", 7)].tensor
    assert HideConstantsPass().apply(graph) is None
    assert set(graph.nodes) == {"x", "a", "r"}
    assert dict(graph.edges) == visible_edges
    assert graph.nodes["a"].input_ports == (0, 7)
    annotation = graph.nodes["a"].visualization["hidden_constants"][7]
    assert annotation == ConstantInput(7, "w", 0, metadata)
    assert annotation.tensor is metadata
    assert str(annotation) == "[7]: fp32[1024, 768]"
    expected = original_nodes["a"]
    expected.visualization["hidden_constants"] = {7: annotation}
    assert graph.nodes["a"] == expected
    assert graph.nodes["x"] == original_nodes["x"]
    assert graph.nodes["r"] == original_nodes["r"]
    graph.validate()


def test_shared_constant_marks_every_consumer_with_same_unique_id(graph):
    graph.add_node(Node("b", "other", "Add", input_ports=(1,)))
    graph.add_edge(Edge("w", 0, "b", 1, graph.edges[("a", 7)].tensor))
    Pipeline([HideConstantsPass()]).apply(graph)
    for node_id, port in (("a", 7), ("b", 1)):
        annotation = graph.nodes[node_id].visualization["hidden_constants"][port]
        assert annotation.shared
        assert annotation.constant_id == "w"
        assert str(annotation) == f"[{port}]: fp32[1024, 768] (shared: w)"
    assert "w" not in graph.nodes


def test_same_consumer_on_two_ports_has_two_records_without_shared_marker():
    graph = Graph(
        [
            Node("c", "constant", "Constant", output_ports=(0,)),
            Node("a", "add", "Add", input_ports=(0, 1)),
        ],
        [Edge("c", 0, "a", 0), Edge("c", 0, "a", 1)],
    )
    Pipeline([HideConstantsPass()]).apply(graph)
    records = graph.nodes["a"].visualization["hidden_constants"]
    assert set(records) == {0, 1}
    assert all(not record.shared for record in records.values())
    assert [record.input_port for record in records.values()] == [0, 1]


def test_constants_with_same_name_keep_separate_ids_and_metadata(graph):
    graph.add_node(Node("w2", "weights", "Constant", output_ports=(3,)))
    graph.remove_edge("a", 0)
    graph.add_edge(Edge("w2", 3, "a", 0, TensorMetadata(TensorShape(()), "i64")))
    HideConstantsPass().apply(graph)
    records = graph.nodes["a"].visualization["hidden_constants"]
    assert records[0].constant_id == "w2" and records[0].output_port == 3
    assert records[7].constant_id == "w"
    assert str(records[0]) == "[0]: i64[]"
    assert str(records[7]) == "[7]: fp32[1024, 768]"
    graph.validate()


@pytest.mark.parametrize(
    "tensor,expected",
    [
        (TensorMetadata(), "?[unknown]"),
        (TensorMetadata(TensorShape(None), "fp32"), "fp32[...]"),
        (TensorMetadata(TensorShape(()), "i64"), "i64[]"),
        (TensorMetadata(TensorShape((None, 3)), "fp16"), "fp16[?, 3]"),
        (TensorMetadata(TensorShape((Dimension(1, 8), 0))), "?[1..8, 0]"),
    ],
)
def test_constant_directly_feeding_result_preserves_metadata(tensor, expected):
    graph = Graph(
        [
            Node("c", "constant", "Constant", output_ports=(0,)),
            Node("r", "output", "Result", input_ports=(0,)),
        ],
        [Edge("c", 0, "r", 0, tensor)],
    )
    Pipeline([HideConstantsPass()]).apply(graph)
    assert set(graph.nodes) == {"r"}
    assert not graph.edges
    assert str(graph.nodes["r"].visualization["hidden_constants"][0]) == f"[0]: {expected}"


def test_constant_with_remaining_visible_input_is_not_removed():
    graph = Graph(
        [
            Node("x", "input", "Custom", output_ports=(0,)),
            Node("c", "constant", "Constant", input_ports=(0,), output_ports=(0,)),
            Node("r", "output", "Result", input_ports=(0,)),
        ],
        [Edge("x", 0, "c", 0), Edge("c", 0, "r", 0)],
    )
    Pipeline([HideConstantsPass()]).apply(graph)
    assert "c" in graph.nodes
    assert tuple(graph.edges.values()) == (Edge("x", 0, "c", 0),)
    assert graph.nodes["r"].visualization["hidden_constants"][0].constant_id == "c"


def test_isolated_constant_is_removed_without_deleting_isolated_other_nodes():
    graph = Graph([Node("c", "unused", "Constant"), Node("a", "unused", "Custom")])
    Pipeline([HideConstantsPass()]).apply(graph)
    assert set(graph.nodes) == {"a"}


@pytest.mark.parametrize("with_nodes", [False, True])
def test_no_constants_is_a_noop(with_nodes):
    graph = Graph([Node("a", "Constant", "Custom")] if with_nodes else [])
    before = deepcopy(dict(graph.nodes))
    assert Pipeline([HideConstantsPass()]).apply(graph) is graph
    assert dict(graph.nodes) == before


def test_repeated_runs_do_not_duplicate_metadata(graph):
    transform = HideConstantsPass()
    transform.apply(graph)
    before_nodes = deepcopy(dict(graph.nodes))
    before_edges = dict(graph.edges)
    Pipeline([transform, transform]).apply(graph)
    assert dict(graph.nodes) == before_nodes
    assert dict(graph.edges) == before_edges


@pytest.mark.parametrize("reverse", [False, True])
def test_composes_with_highlighting_and_keeps_selected_consumer(graph, reverse):
    steps = [HighlightPass(["matmul", "weights"]), HideConstantsPass()]
    Pipeline(reversed(steps) if reverse else steps).apply(graph)
    assert graph.nodes["a"].visualization["border_color"] == "red"
    assert "w" not in graph.nodes
    assert 7 in graph.nodes["a"].visualization["hidden_constants"]


def test_disabled_constant_hiding_changes_nothing(graph):
    nodes, edges = deepcopy(dict(graph.nodes)), dict(graph.edges)
    Pipeline([PassStep(HideConstantsPass(), enabled=False)]).apply(graph)
    assert dict(graph.nodes) == nodes
    assert dict(graph.edges) == edges


def test_constant_attributes_including_values_are_not_inspected_or_copied(graph):
    class Forbidden:
        def __str__(self):
            raise AssertionError("Tensor values must not be displayed")

        def __deepcopy__(self, memo):
            raise AssertionError("Tensor values must not be copied")

    graph.nodes["w"].attributes["value"] = Forbidden()
    Pipeline([HideConstantsPass()]).apply(graph)
    assert str(graph.nodes["a"].visualization["hidden_constants"][7]) == "[7]: fp32[1024, 768]"


@pytest.mark.parametrize(
    "existing",
    ["custom data", {7: "custom"}, {7: ConstantInput(7, "previous", 0, TensorMetadata())}],
)
def test_metadata_conflicts_are_rejected_before_any_changes(graph, existing):
    # The first consumer would be modified before this conflict without a planning phase.
    graph.add_node(
        Node("b", "other", "Custom", input_ports=(7,), visualization={"hidden_constants": existing})
    )
    graph.add_edge(Edge("w", 0, "b", 7))
    before_nodes, before_edges = deepcopy(dict(graph.nodes)), dict(graph.edges)
    with pytest.raises(GraphValidationError, match="metadata"):
        HideConstantsPass().apply(graph)
    assert dict(graph.nodes) == before_nodes
    assert dict(graph.edges) == before_edges
    graph.validate()


def test_existing_annotations_on_other_ports_are_preserved(graph):
    previous = ConstantInput(0, "previous", 0, TensorMetadata())
    graph.remove_edge("a", 0)
    graph.nodes["a"].visualization["hidden_constants"] = {0: previous}
    Pipeline([HideConstantsPass()]).apply(graph)
    records = graph.nodes["a"].visualization["hidden_constants"]
    assert records[0] is previous
    assert set(records) == {0, 7}
