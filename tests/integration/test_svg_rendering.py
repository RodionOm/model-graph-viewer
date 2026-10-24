"""Semantic checks on real Graphviz SVG, including an OpenVINO IR round trip."""

import copy
from xml.etree import ElementTree as ET

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import Dimension, Edge, Graph, Node, TensorMetadata, TensorShape
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline
from model_graph_viewer.renderers import GraphvizRenderer

NS = {"s": "http://www.w3.org/2000/svg"}


def groups(root, kind):
    return root.findall(f".//s:g[@class='{kind}']", NS)


def texts(root):
    return [item.text or "" for item in root.findall(".//s:text", NS)]


@pytest.mark.parametrize("layout", ["default", "fast"])
def test_real_ir_to_transformed_svg_preserves_model_and_styles(tmp_path, layout):
    x = ops.parameter([1, 2], ov.Type.f32, name="input")
    w = ops.constant(np.array([[0.5], [1.0]], dtype=np.float32), name="weights")
    b = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    y = ops.add(ops.matmul(x, w, False, False, name="matmul"), b, name="add")
    model = ov.Model([ops.result(y, name="output")], [x])
    path = tmp_path / "model.xml"
    ov.serialize(model, path)
    original = (path.read_bytes(), path.with_suffix(".bin").read_bytes())
    loader = OpenVINOLoader()
    loaded = loader.read_model(path)
    graph = loader.parse(loaded)
    renderer = GraphvizRenderer(layout=layout)
    before = ET.fromstring(renderer.render_svg(graph))
    assert len(groups(before, "node")) == 6
    assert len(groups(before, "edge")) == 5
    Pipeline([HighlightPass(["matmul"]), HideConstantsPass()]).apply(graph)
    snapshot = copy.deepcopy((dict(graph.nodes), dict(graph.edges)))
    output = renderer.render(graph, tmp_path / "result.svg")
    after = ET.parse(output).getroot()
    assert len(groups(after, "node")) == 4
    assert len(groups(after, "edge")) == 3
    labels = texts(after)
    assert {
        "input",
        "matmul",
        "add",
        "output",
        "transpose_a: False",
        "transpose_b: False",
        "[1]: fp32[2, 1]",
        "[1]: fp32[1]",
    } <= set(labels)
    assert "weights" not in labels
    assert "bias" not in labels
    assert "fp32[1, 2]" in labels
    matmul = next(n for n in groups(after, "node") if "matmul" in texts(n))
    assert matmul.find(".//s:polygon[@stroke='red']", NS) is not None
    assert matmul.find(".//s:polygon[@fill='#dbeafe']", NS) is not None
    assert all("out 0 → in 0" in texts(e) for e in groups(after, "edge"))
    assert (dict(graph.nodes), dict(graph.edges)) == snapshot
    ov.serialize(loaded, tmp_path / "unchanged.xml")
    assert (path.read_bytes(), path.with_suffix(".bin").read_bytes()) == original
    assert (
        (tmp_path / "unchanged.xml").read_bytes(),
        (tmp_path / "unchanged.bin").read_bytes(),
    ) == original


def test_parallel_edges_shared_constant_and_multiple_outputs():
    x = ops.parameter([2, 4], ov.Type.f32, name="input")
    split = ops.split(x, ops.constant(np.int64(1), name="axis"), 2, name="split")
    shared = ops.constant(np.array([1, 2], dtype=np.float32), name="shared")
    merge = ops.add(split.output(0), split.output(1), name="merge")
    left = ops.add(merge, shared, name="left")
    right = ops.add(split.output(1), shared, name="right")
    graph = OpenVINOLoader().parse(ov.Model([left, right], [x]))
    shared_id = next(n.node_id for n in graph.nodes.values() if n.name == "shared")
    Pipeline([HideConstantsPass(), HighlightPass(["split"])]).apply(graph)
    root = ET.fromstring(GraphvizRenderer().render_svg(graph))
    assert len(groups(root, "node")) == len(graph.nodes)
    assert len(groups(root, "edge")) == len(graph.edges)
    assert texts(root).count(f"[1]: fp32[2] (shared: {shared_id})") == 2
    assert "[1]: i64[]" in texts(root)
    assert "out 1 → in 1" in texts(root)
    merge_group = next(n for n in groups(root, "node") if "merge" in texts(n))
    merge_id = merge_group.find("s:title", NS).text
    incoming = [e for e in groups(root, "edge") if f"->{merge_id}:" in e.find("s:title", NS).text]
    assert len(incoming) == 2
    assert any("out 0 → in 0" in texts(edge) for edge in incoming)
    assert any("out 1 → in 1" in texts(edge) for edge in incoming)


def test_special_characters_remain_literal_svg_text():
    name = "<script>alert(\"x\")</script> & 'quoted' \\N \\n \\l → нейрон"
    graph = Graph(
        [
            Node(
                'id:"odd',
                name,
                "Unknown<Op>",
                attributes={
                    "details": "first\nsecond\tvalue\x01",
                },
                visualization={"note": "body <not expanded>"},
            ),
        ]
    )
    root = ET.fromstring(GraphvizRenderer().render_svg(graph))
    labels = texts(root)
    assert name in labels
    assert 'ID: id:"odd' in labels
    assert "Unknown<Op>" in labels
    assert "details: first" in labels
    assert "second    value�" in [text.replace("\u00a0", " ") for text in labels]
    assert "body <not expanded>" in labels
    assert not any(element.tag.endswith("script") for element in root.iter())
    assert root.find(".//s:polygon[@fill='#f8fafc']", NS) is not None


@pytest.mark.parametrize(
    "tensor",
    [
        TensorMetadata(),
        TensorMetadata(TensorShape(None), "fp32"),
        TensorMetadata(TensorShape(()), "i64"),
        TensorMetadata(TensorShape((Dimension(1, 8), None, 0)), "fp16"),
    ],
)
def test_tensor_labels_and_arbitrary_port_numbers(tensor):
    graph = Graph(
        [
            Node("a", "same", "Custom", output_ports=(2, 5)),
            Node("b", "same", "Custom", input_ports=(1, 4, 7)),
        ],
        [Edge("a", 5, "b", 7, tensor)],
    )
    root = ET.fromstring(GraphvizRenderer().render_svg(graph))
    assert str(tensor) in texts(root)
    assert "out 5 → in 7" in texts(root)
    assert texts(root).count("same") == 2
    # Graphviz's SVG title retains compass directions, but omits HTML port names.
    assert groups(root, "edge")[0].find("s:title", NS).text == "n0:s->n1:n"


@pytest.mark.parametrize("nodes", [[], [Node("only", "isolated", "Custom")]])
def test_empty_and_single_node_graphs_are_valid_svg(nodes):
    root = ET.fromstring(GraphvizRenderer().render_svg(Graph(nodes)))
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(groups(root, "node")) == len(nodes)
    assert not groups(root, "edge")
