"""Connectivity, labels and export contracts for readable large graphs."""

import copy
import hashlib
import json

import pytest

from model_graph_viewer import Dimension, Edge, Graph, Node, TensorMetadata, TensorShape, api, cli
from model_graph_viewer.configuration import configured_passes
from model_graph_viewer.errors import ConfigurationError, RenderError
from model_graph_viewer.passes import (
    FanoutReferencesPass,
    HideConstantsPass,
    HighlightPass,
    Pipeline,
)
from model_graph_viewer.renderers import GraphvizRenderer


def fanout_graph(count=4):
    tensor = TensorMetadata(TensorShape((Dimension(1, 8), None)), "fp32")
    return Graph(
        [
            Node(
                "source:<>",
                "duplicate",
                "Custom",
                output_ports=(0, 7),
                attributes={"long": "untruncated " * 80},
            )
        ]
        + [
            Node(str(i), "duplicate", "Add", input_ports=(2,), output_ports=(0,))
            for i in range(count)
        ],
        [Edge("source:<>", 0 if i % 2 else 7, str(i), 2, tensor) for i in range(count)],
    )


@pytest.mark.parametrize("count,selected", [(0, False), (3, False), (4, True)])
def test_threshold_is_strict_and_pass_never_changes_connectivity(count, selected):
    graph = fanout_graph(count)
    before = dict(graph.edges)
    Pipeline([HighlightPass(["duplicate"]), FanoutReferencesPass()]).apply(graph)
    assert graph.nodes["source:<>"].visualization.get("reference_fanout", False) is selected
    assert dict(graph.edges) == before
    assert all(n.visualization["border_color"] == "red" for n in graph.nodes.values())
    assert len(graph.nodes) == count + 1
    snapshot = copy.deepcopy(dict(graph.nodes))
    Pipeline([FanoutReferencesPass()]).apply(graph)
    assert dict(graph.nodes) == snapshot


@pytest.mark.parametrize("threshold", [0, -1, True, 2.5, "3"])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="positive integer"):
        FanoutReferencesPass(threshold)


def test_parallel_edges_and_self_loop_are_counted_without_losing_ports():
    graph = Graph(
        [
            Node("s", "same", "Custom", input_ports=(0,), output_ports=(4,)),
            Node("t", "same", "Add", input_ports=(0, 1, 2)),
        ],
        [Edge("s", 4, "s", 0)] + [Edge("s", 4, "t", i) for i in range(3)],
    )
    FanoutReferencesPass().apply(graph)
    dot = GraphvizRenderer().to_dot(graph).source
    assert " -> " not in dot
    for port in range(3):
        assert f"in {port} &#8592; #s Custom : out 4" in dot
    assert "4 reference(s)" in dot
    graph.remove_edge("s", 0)
    FanoutReferencesPass().apply(graph)
    assert "reference_fanout" not in graph.nodes["s"].visualization
    assert GraphvizRenderer().to_dot(graph).source.count(" -> ") == 3


def test_compact_labels_full_metadata_and_render_are_nonmutating():
    graph = fanout_graph()
    FanoutReferencesPass().apply(graph)
    before = copy.deepcopy((dict(graph.nodes), dict(graph.edges)))
    renderer = GraphvizRenderer(compact=True)
    dot = renderer.to_dot(graph).source
    assert "long:" not in dot
    assert "duplicate<BR" not in dot
    assert 'HREF="#node_n0"' in dot
    assert "#source:&lt;&gt; Custom : out 7" in dot
    assert "fp32[1..8, ?]" in dot
    assert " -> " not in dot
    metadata = json.loads(json.dumps(renderer.metadata(graph)))
    assert metadata["nodes"]["source:<>"]["attributes"]["long"] == "untruncated " * 80
    assert metadata["nodes"]["source:<>"]["name"] == "duplicate"
    assert len(metadata["edges"]) == 4
    assert not any(e["rendered"] for e in metadata["edges"])
    assert metadata["edges"][0]["tensor"]["shape"]["dimensions"] == [
        {"lower": 1, "upper": 8},
        None,
    ]
    assert (dict(graph.nodes), dict(graph.edges)) == before


def test_hide_constants_then_references_preserves_hidden_and_visible_connections():
    graph = fanout_graph()
    graph.add_node(Node("c", "weights", "Constant", output_ports=(0,)))
    graph.add_node(Node("consumer", "consumer", "Add", input_ports=(0, 1)))
    graph.add_edge(Edge("c", 0, "consumer", 1))
    graph.add_edge(Edge("source:<>", 7, "consumer", 0))
    Pipeline([HideConstantsPass(), FanoutReferencesPass()]).apply(graph)
    renderer = GraphvizRenderer(compact=True)
    renderer.to_dot(graph)
    data = renderer.metadata(graph)
    assert len(data["edges"]) == 5
    assert data["nodes"]["consumer"]["hidden_constants"][0]["constant_id"] == "c"
    assert data["nodes"]["consumer"]["hidden_constants"][0]["input_port"] == 1


def test_metadata_never_reads_constant_values_or_opaque_arrays():
    class Tensor:
        def __str__(self):
            raise AssertionError("must not read weights")

        __repr__ = __str__

    cycle = []
    cycle.append(cycle)
    graph = Graph(
        [
            Node("c", "constant", "Constant", attributes={"value": Tensor(), "dtype": "fp16"}),
            Node("a", "other", "Unknown", attributes={"array": Tensor(), "cycle": cycle}),
        ]
    )
    data = GraphvizRenderer(compact=True).metadata(graph)
    assert data["nodes"]["c"]["attributes"] == {"dtype": "fp16"}
    assert data["nodes"]["a"]["attributes"]["array"] == {"unavailable_type": "Tensor"}
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize(
    "shape", [None, TensorShape(None), TensorShape(()), TensorShape((0, None))]
)
def test_metadata_retains_distinct_shapes(shape):
    graph = fanout_graph(1)
    graph.remove_edge("0", 2)
    graph.add_edge(Edge("source:<>", 7, "0", 2, TensorMetadata(shape, None)))
    tensor = GraphvizRenderer().metadata(graph)["edges"][0]["tensor"]
    assert tensor["dtype"] is None
    if shape is None:
        assert tensor["shape"] is None
    else:
        assert tensor["shape"]["dimensions"] == shape.dimensions


def test_sidecar_is_automatic_and_paired_with_svg_hash(monkeypatch, tmp_path):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    monkeypatch.setattr(GraphvizRenderer, "render_svg", lambda *a: svg)
    graph = fanout_graph()
    FanoutReferencesPass().apply(graph)
    output = tmp_path / "graph.svg"
    GraphvizRenderer().render(graph, output)
    data = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["svg_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(data["edges"]) == len(graph.edges)


def test_failed_render_preserves_existing_pair(monkeypatch, tmp_path):
    output = tmp_path / "graph.svg"
    sidecar = output.with_suffix(".json")
    output.write_bytes(b"previous svg")
    sidecar.write_bytes(b"previous json")

    def fail(*args):
        raise RenderError("layout failed")

    monkeypatch.setattr(GraphvizRenderer, "render_svg", fail)
    with pytest.raises(RenderError):
        GraphvizRenderer(compact=True).render(fanout_graph(), output)
    assert output.read_bytes() == b"previous svg"
    assert sidecar.read_bytes() == b"previous json"


def test_readability_cli_and_config_options(monkeypatch, tmp_path):
    def visualize(model, output, **kwargs):
        assert kwargs["compact"] is True
        assert kwargs["passes"][1].threshold == 5
        assert kwargs["metadata_path"].name == "details.json"
        return output

    monkeypatch.setattr(cli, "visualize", visualize)
    assert (
        cli.main(
            [
                "--model",
                "m.xml",
                "--output",
                "o.svg",
                "--compact",
                "--passes",
                "hide-constants,fanout-references",
                "--fanout-threshold",
                "5",
                "--metadata-output",
                "details.json",
            ]
        )
        == 0
    )
    config = tmp_path / "passes.json"
    config.write_text('[{"name":"fanout-references","options":{"threshold":7}}]')
    assert configured_passes(config)[0].transformation.threshold == 7


@pytest.mark.parametrize(
    "options",
    [
        ["--fanout-threshold", "0"],
        ["--fanout-threshold", "3"],
        ["--metadata-output", "bad.svg"],
        ["--check-environment", "--compact"],
        ["--pass-config", "x.json", "--fanout-threshold", "3"],
    ],
)
def test_invalid_readability_cli_options(options):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", "m.xml", "--output", "o.svg", *options])
    assert exc.value.code == 2


def test_json_output_cannot_overwrite_custom_input():
    with pytest.raises(ConfigurationError, match="overwrite"):
        api.visualize("input.json", "output.svg", metadata_path="input.json")


def test_compact_boundary_names_are_short_and_other_names_are_hidden():
    renderer = GraphvizRenderer(compact=True)
    for op_type in ("Parameter", "Result"):
        node = Node("p", "attention_mask", op_type)
        assert "attention_mask<BR/>" in renderer._label(node)
        long = Node("r", "output_" * 30, op_type)
        assert long.name[:29] + "...<BR/>" in renderer._label(long)
        assert renderer.metadata(Graph([long]))["nodes"]["r"]["name"] == long.name
    assert "attention_mask" not in renderer._label(Node("m", "attention_mask", "MatMul"))


def test_compact_group_hint_only_marks_visible_add_to_add_endpoints():
    graph = Graph(
        [
            Node("a", "a", "Add", output_ports=(0,)),
            Node("b", "b", "Add", input_ports=(0,), output_ports=(0,)),
            Node("c", "c", "MatMul", input_ports=(0,)),
        ],
        [Edge("a", 0, "b", 0), Edge("b", 0, "c", 0)],
    )
    before = copy.deepcopy((dict(graph.nodes), dict(graph.edges)))
    source = GraphvizRenderer(compact=True).to_dot(graph).source
    assert source.count("group=residual_backbone") == 2
    assert "weight=10" not in source
    assert "group=residual_backbone" not in GraphvizRenderer().to_dot(graph).source
    assert (dict(graph.nodes), dict(graph.edges)) == before
    graph.nodes["a"].visualization["reference_fanout"] = True
    assert "group=residual_backbone" not in GraphvizRenderer(compact=True).to_dot(graph).source


@pytest.mark.parametrize(
    "name, expected",
    [
        ("beam_idx", "beam_idx"),
        ("attention_mask", "attention_mask"),
        ("long_input_" * 8, ("long_input_" * 8)[:29] + "..."),
        ("input<&>", "input&lt;&amp;&gt;"),
        ("", "#input Parameter"),
    ],
)
def test_parameter_references_use_bounded_names_and_retain_identity(name, expected):
    graph = Graph(
        [
            Node("input", name, "Parameter", output_ports=(7,)),
            Node("a", "consumer", "Add", input_ports=(0, 1)),
        ],
        [Edge("input", 7, "a", p) for p in (0, 1)],
    )
    FanoutReferencesPass(threshold=1).apply(graph)
    renderer = GraphvizRenderer(compact=True)
    dot = renderer.to_dot(graph).source
    for port in (0, 1):
        assert f"in {port} &#8592; {expected} : out 7" in dot
    assert dot.count('HREF="#node_n0"') == 2
    assert renderer.metadata(graph)["nodes"]["input"]["name"] == name
    assert {e["from_node"] for e in renderer.metadata(graph)["edges"]} == {"input"}
