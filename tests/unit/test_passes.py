import subprocess
import sys
from copy import deepcopy

import pytest

from model_graph_viewer import Edge, Graph, Node, TensorMetadata, TensorShape
from model_graph_viewer.errors import GraphValidationError
from model_graph_viewer.passes import HighlightPass, PassStep, Pipeline


@pytest.fixture
def graph():
    return Graph(
        [
            Node(
                "0",
                "target",
                "Parameter",
                output_ports=(0,),
                attributes={"shape": [1, 2]},
                visualization={"fillcolor": "blue"},
            ),
            Node(
                "1",
                "target",
                "Add",
                input_ports=(0, 1),
                output_ports=(0,),
                visualization={"border_color": "green", "note": "keep me"},
            ),
            Node(
                "target",
                "other",
                "Result",
                input_ports=(0,),
                visualization={"border_color": "purple"},
            ),
        ],
        [
            Edge("0", 0, "1", 0, TensorMetadata(TensorShape((1, 2)), "fp32")),
            Edge("0", 0, "1", 1),
            Edge("1", 0, "target", 0),
        ],
    )


def test_highlight_changes_only_matching_border_metadata(graph):
    before_nodes = deepcopy(dict(graph.nodes))
    before_edges = dict(graph.edges)
    assert HighlightPass(["target"]).apply(graph) is None
    assert graph.nodes["0"].visualization == {"fillcolor": "blue", "border_color": "red"}
    assert graph.nodes["1"].visualization == {"border_color": "red", "note": "keep me"}
    assert graph.nodes["target"] == before_nodes["target"]
    for node_id, node in graph.nodes.items():
        expected = before_nodes[node_id]
        if expected.name == "target":
            expected.visualization["border_color"] = "red"
        assert node == expected
    assert dict(graph.edges) == before_edges
    graph.validate()


@pytest.mark.parametrize("names", [[], ["missing"], ["Target"], ["tar"], ["target "]])
def test_unmatched_selection_does_not_change_graph(graph, names):
    before_nodes = deepcopy(dict(graph.nodes))
    before_edges = dict(graph.edges)
    Pipeline([HighlightPass(names)]).apply(graph)
    assert dict(graph.nodes) == before_nodes
    assert dict(graph.edges) == before_edges


def test_names_with_spaces_commas_and_unicode_are_literal():
    name = "layer, \u03b1 output"
    graph = Graph([Node("0", name, "Custom"), Node("1", "layer", "Custom")])
    HighlightPass([name]).apply(graph)
    assert graph.nodes["0"].visualization == {"border_color": "red"}
    assert graph.nodes["1"].visualization == {}


def test_highlight_is_repeatable_and_keeps_previous_selections(graph):
    highlight = HighlightPass(["target", "target"])
    highlight.apply(graph)
    once = deepcopy(dict(graph.nodes))
    highlight.apply(graph)
    assert dict(graph.nodes) == once
    HighlightPass(["other"]).apply(graph)
    assert all(n.visualization["border_color"] == "red" for n in graph.nodes.values())


def test_name_configuration_is_snapshotted_and_generator_is_reusable(graph):
    names = ["target"]
    highlight = HighlightPass(iter(names))
    names.clear()
    highlight.apply(graph)
    assert graph.nodes["0"].visualization["border_color"] == "red"
    another = Graph([Node("new", "target", "Custom")])
    highlight.apply(another)
    assert another.nodes["new"].visualization["border_color"] == "red"


@pytest.mark.parametrize("names", ["target", [1], [None], [[]]])
def test_invalid_name_configuration_is_rejected(names):
    with pytest.raises(TypeError, match="name|names"):
        HighlightPass(names)


def test_empty_graph_and_empty_pipeline():
    graph = Graph()
    assert Pipeline().apply(graph) is graph
    assert Pipeline([HighlightPass(["unknown"])]).apply(graph) is graph
    assert not graph.nodes and not graph.edges


def test_empty_pipeline_preserves_nonempty_graph(graph):
    before_nodes = deepcopy(dict(graph.nodes))
    before_edges = dict(graph.edges)
    assert Pipeline().apply(graph) is graph
    assert dict(graph.nodes) == before_nodes
    assert dict(graph.edges) == before_edges


class AddTarget:
    def apply(self, graph):
        graph.add_node(Node("created", "target", "Custom"))


def test_dependent_passes_observe_the_requested_order():
    first = Graph()
    Pipeline([AddTarget(), HighlightPass(["target"])]).apply(first)
    assert first.nodes["created"].visualization == {"border_color": "red"}
    reversed_graph = Graph()
    Pipeline([HighlightPass(["target"]), AddTarget()]).apply(reversed_graph)
    assert reversed_graph.nodes["created"].visualization == {}


def test_disabled_step_is_never_called(graph):
    class MustNotRun:
        def apply(self, graph):
            raise AssertionError("Disabled pass was called")

    pipeline = Pipeline(
        [
            PassStep(MustNotRun(), enabled=False),
            HighlightPass(["target"]),
            PassStep(HighlightPass(["other"]), enabled=False),
        ]
    )
    pipeline.apply(graph)
    assert graph.nodes["0"].visualization["border_color"] == "red"
    assert graph.nodes["target"].visualization["border_color"] == "purple"


def test_enabled_step_can_be_selected_explicitly(graph):
    Pipeline([PassStep(HighlightPass(["other"]), enabled=True)]).apply(graph)
    assert graph.nodes["target"].visualization["border_color"] == "red"
    assert graph.nodes["0"].visualization == {"fillcolor": "blue"}


def test_pipeline_keeps_same_graph_and_nodes_without_copying(graph):
    node = graph.nodes["0"]
    edge = graph.edges[("1", 0)]
    result = Pipeline([HighlightPass(["target"])]).apply(graph)
    assert result is graph
    assert result.nodes["0"] is node
    assert result.edges[("1", 0)] is edge


def test_sequence_is_snapshotted_and_repeated_passes_are_not_deduplicated(graph):
    class Increment:
        def apply(self, graph):
            data = graph.nodes["0"].visualization
            data["count"] = data.get("count", 0) + 1

    increment = Increment()
    steps = [increment, increment]
    pipeline = Pipeline(iter(steps))
    steps.clear()
    pipeline.apply(graph)
    assert graph.nodes["0"].visualization["count"] == 2
    pipeline.apply(graph)
    assert graph.nodes["0"].visualization["count"] == 4


def test_invalid_input_graph_is_rejected_before_any_pass(graph):
    graph._outgoing["0"].clear()
    with pytest.raises(GraphValidationError, match="Outgoing edge index"):
        Pipeline([HighlightPass(["target"])]).apply(graph)
    assert graph.nodes["0"].visualization == {"fillcolor": "blue"}


def test_invalid_pass_output_stops_pipeline_with_context(graph):
    class BreakIndex:
        def apply(self, graph):
            graph._outgoing["0"].clear()

    with pytest.raises(GraphValidationError, match=r"Pass 2 \(BreakIndex\)") as error:
        Pipeline([HighlightPass(["target"]), BreakIndex(), HighlightPass(["other"])]).apply(graph)
    assert isinstance(error.value.__cause__, GraphValidationError)
    assert graph.nodes["0"].visualization["border_color"] == "red"
    assert graph.nodes["target"].visualization["border_color"] == "purple"


def test_programming_error_propagates_and_stops_later_passes(graph):
    class BrokenPass:
        def apply(self, graph):
            raise RuntimeError("custom bug")

    with pytest.raises(RuntimeError, match="custom bug"):
        Pipeline([BrokenPass(), HighlightPass(["target"])]).apply(graph)
    assert graph.nodes["0"].visualization == {"fillcolor": "blue"}


def test_returning_another_graph_violates_the_in_place_contract(graph):
    class ReplacingPass:
        def apply(self, graph):
            return Graph()

    with pytest.raises(TypeError, match="return None"):
        Pipeline([ReplacingPass(), HighlightPass(["target"])]).apply(graph)
    assert graph.nodes["0"].visualization == {"fillcolor": "blue"}


@pytest.mark.parametrize("enabled", ["false", 0, None])
def test_invalid_enabled_flag_is_rejected(enabled):
    with pytest.raises(TypeError, match="boolean"):
        PassStep(HighlightPass([]), enabled=enabled)


def test_invalid_pass_configuration_is_rejected():
    with pytest.raises(TypeError, match="apply"):
        Pipeline([object()])


def test_passes_do_not_import_model_or_rendering_libraries():
    code = (
        "import sys; from model_graph_viewer import Graph; "
        "from model_graph_viewer.passes import Pipeline, HighlightPass; "
        "Pipeline([HighlightPass([])]).apply(Graph()); "
        "assert 'openvino' not in sys.modules; assert 'graphviz' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
