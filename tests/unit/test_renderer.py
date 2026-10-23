"""Renderer contracts and failure handling without the native dot executable."""

import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from model_graph_viewer import Edge, Graph, Node, TensorMetadata, TensorShape
from model_graph_viewer.errors import RenderError
from model_graph_viewer.renderers import GraphvizRenderer
from model_graph_viewer.renderers import graphviz as renderer_module

SVG = b'<svg xmlns="http://www.w3.org/2000/svg"/>'


def test_dot_preserves_distinct_ports_arbitrary_ids_and_graph_data():
    graph = Graph(
        [
            Node('source:"id', "same", "UnknownOp", output_ports=(2, 4)),
            Node("target:other", "same", "Add", input_ports=(0, 7, 8), output_ports=(0,)),
        ],
        [Edge('source:"id', 4, "target:other", p) for p in (0, 7)],
    )
    before = copy.deepcopy((dict(graph.nodes), dict(graph.edges)))
    dot = GraphvizRenderer().to_dot(graph)
    assert not dot.strict
    assert "rankdir=TB" in dot.source
    assert "n0:out4:s -> n1:in0:n" in dot.source
    assert "n0:out4:s -> n1:in7:n" in dot.source
    assert 'PORT="in8"' in dot.source
    assert 'PORT="out2"' in dot.source
    assert (dict(graph.nodes), dict(graph.edges)) == before


def test_shortening_and_opaque_values_never_modify_or_read_original_data():
    class Tensor:
        def __str__(self):
            raise AssertionError("tensor values must not be read")

        __repr__ = __str__

    tensor = Tensor()
    name = "very long name " * 20
    graph = Graph(
        [
            Node("0", name, "Constant", attributes={"values": tensor, "dtype": "fp32"}),
            Node("1", "operation", "Custom", attributes={"opaque": tensor, "text": name}),
        ]
    )
    source = GraphvizRenderer(max_label_length=24).to_dot(graph).source
    assert "values" not in source
    assert "&lt;Tensor&gt;" in source
    assert "very long name very l..." in source
    assert name not in source
    assert graph.nodes["0"].name == name
    assert graph.nodes["0"].attributes["values"] is tensor
    assert graph.nodes["1"].attributes["text"] == name


@pytest.mark.parametrize(
    "options",
    [
        {"max_label_length": 7},
        {"max_label_length": True},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": True},
    ],
)
def test_invalid_renderer_options(options):
    with pytest.raises(ValueError):
        GraphvizRenderer(**options)


@pytest.mark.parametrize(
    "metadata",
    [
        {"border_color": 5},
        {"fillcolor": None},
        {"hidden_constants": []},
        {"hidden_constants": {0: "invalid"}},
    ],
)
def test_invalid_visualization_metadata_is_reported(metadata):
    graph = Graph([Node("0", "x", "Custom", input_ports=(0,), visualization=metadata)])
    with pytest.raises(RenderError, match="Node '0'"):
        GraphvizRenderer().to_dot(graph)


def test_missing_dot_gives_installation_hint(monkeypatch):
    monkeypatch.setattr(renderer_module.shutil, "which", lambda _: None)
    with pytest.raises(RenderError, match="add its bin directory to PATH"):
        GraphvizRenderer().render_svg(Graph())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.CalledProcessError(1, "dot", stderr=b"syntax error"), "syntax error"),
        (subprocess.TimeoutExpired("dot", 2), "2-second timeout"),
        (OSError("permission denied"), "Cannot execute Graphviz"),
    ],
)
def test_dot_errors_are_contextual(monkeypatch, failure, message):
    monkeypatch.setattr(renderer_module.shutil, "which", lambda _: "dot")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(renderer_module.subprocess, "run", fail)
    with pytest.raises(RenderError, match=message):
        GraphvizRenderer(timeout=2).render_svg(Graph())


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"<svg", "invalid XML"),
        (b"<html/>", "not an SVG"),
    ],
)
def test_invalid_dot_output_does_not_replace_existing_svg(monkeypatch, tmp_path, data, message):
    output = tmp_path / "output.svg"
    output.write_bytes(SVG)
    monkeypatch.setattr(renderer_module.shutil, "which", lambda _: "dot")
    monkeypatch.setattr(
        renderer_module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=data, stderr=b"")
    )
    with pytest.raises(RenderError, match=message):
        GraphvizRenderer().render(Graph(), output)
    assert output.read_bytes() == SVG
    assert list(tmp_path.iterdir()) == [output]


def test_render_uses_stdin_and_surfaces_warnings(monkeypatch, caplog):
    monkeypatch.setattr(renderer_module.shutil, "which", lambda _: "/path with spaces/dot")

    def run(command, **kwargs):
        assert command == ["/path with spaces/dot", "-Tsvg"]
        assert kwargs["input"].startswith(b"digraph")
        assert kwargs["timeout"] == 3
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        return SimpleNamespace(stdout=SVG, stderr=b"layout warning")

    monkeypatch.setattr(renderer_module.subprocess, "run", run)
    assert GraphvizRenderer(timeout=3).render_svg(Graph()) == SVG
    assert "layout warning" in caplog.text


def test_atomic_output_creates_parent_and_replaces_file(monkeypatch, tmp_path):
    monkeypatch.setattr(GraphvizRenderer, "render_svg", lambda *a: SVG)
    output = tmp_path / "folder with spaces" / "граф.svg"
    assert GraphvizRenderer().render(Graph(), output) == output
    output.write_bytes(b"old")
    GraphvizRenderer().render(Graph(), output)
    assert output.read_bytes() == SVG
    assert list(output.parent.iterdir()) == [output]


def test_failed_replace_keeps_original_and_cleans_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(GraphvizRenderer, "render_svg", lambda *a: SVG)
    output = tmp_path / "graph.svg"
    output.write_bytes(b"previous content")

    def fail(*args):
        raise PermissionError("destination locked")

    monkeypatch.setattr(renderer_module.os, "replace", fail)
    with pytest.raises(RenderError, match="Cannot write SVG.*destination locked"):
        GraphvizRenderer().render(Graph(), output)
    assert output.read_bytes() == b"previous content"
    assert list(tmp_path.iterdir()) == [output]


def test_unwritable_parent_reports_path(monkeypatch, tmp_path):
    monkeypatch.setattr(GraphvizRenderer, "render_svg", lambda *a: SVG)
    parent = tmp_path / "regular-file"
    parent.write_text("keep")
    with pytest.raises(RenderError, match="regular-file"):
        GraphvizRenderer().render(Graph(), parent / "graph.svg")
    assert parent.read_text() == "keep"


def test_invalid_output_suffix_is_rejected_before_render(monkeypatch, tmp_path):
    def unexpected(*args):
        raise AssertionError("must validate path before rendering")

    monkeypatch.setattr(GraphvizRenderer, "render_svg", unexpected)
    with pytest.raises(RenderError, match="must end with .svg"):
        GraphvizRenderer().render(Graph(), tmp_path / "graph.png")


def test_renderer_import_does_not_load_openvino():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from model_graph_viewer.renderers import GraphvizRenderer; "
            "assert 'openvino' not in sys.modules",
        ],
        check=True,
        capture_output=True,
    )


def test_tensor_metadata_is_not_truncated():
    tensor = TensorMetadata(TensorShape((None, 1024, 2048)), "fp32")
    graph = Graph(
        [
            Node("a", "a", "A", output_ports=(0,)),
            Node("b", "b", "B", input_ports=(0,)),
        ],
        [Edge("a", 0, "b", 0, tensor)],
    )
    assert "fp32[?, 1024, 2048]" in GraphvizRenderer(max_label_length=8).to_dot(graph).source
