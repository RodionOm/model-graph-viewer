import subprocess
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import Dimension, TensorShape
from model_graph_viewer.errors import ModelLoadError
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, PassStep, Pipeline


@pytest.fixture
def linear_model():
    x = ops.parameter([1, 2], ov.Type.f32, name="input")
    w = ops.constant(np.array([[0.5], [1.0]], dtype=np.float32), name="weights")
    b = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    y = ops.add(ops.matmul(x, w, False, False, name="matmul"), b, name="add")
    return ov.Model([ops.result(y, name="output")], [x], "linear")


def write_model(model, directory, name="model"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.xml"
    ov.serialize(model, path)
    return path


def test_load_real_ir_preserves_operations_connections_and_tensor_metadata(linear_model, tmp_path):
    path = write_model(linear_model, tmp_path)
    assert path.with_suffix(".bin").stat().st_size == 12
    graph = OpenVINOLoader().load(path)
    assert {n.name: n.op_type for n in graph.nodes.values()} == {
        "input": "Parameter",
        "weights": "Constant",
        "bias": "Constant",
        "matmul": "MatMul",
        "add": "Add",
        "output": "Result",
    }
    connections = {
        (graph.nodes[e.from_node].name, e.from_port, graph.nodes[e.to_node].name, e.to_port): (
            e.tensor.shape.dimensions,
            e.tensor.dtype,
        )
        for e in graph.edges.values()
    }
    assert connections == {
        ("input", 0, "matmul", 0): ((1, 2), "fp32"),
        ("weights", 0, "matmul", 1): ((2, 1), "fp32"),
        ("matmul", 0, "add", 0): ((1, 1), "fp32"),
        ("bias", 0, "add", 1): ((1,), "fp32"),
        ("add", 0, "output", 0): ((1, 1), "fp32"),
    }
    matmul = next(n for n in graph.nodes.values() if n.op_type == "MatMul")
    assert matmul.attributes == {"transpose_a": False, "transpose_b": False}
    assert matmul.input_ports == (0, 1)
    assert matmul.output_ports == (0,)
    graph.validate()


def test_ids_repeat_on_read_and_do_not_depend_on_friendly_names(linear_model, tmp_path):
    for operation in linear_model.get_ordered_ops():
        operation.set_friendly_name("duplicate")
    path = write_model(linear_model, tmp_path)
    first = OpenVINOLoader().load(path)
    second = OpenVINOLoader().load(path)
    assert len(first.nodes) == 6
    assert all(n.name == "duplicate" for n in first.nodes.values())
    assert dict(first.nodes) == dict(second.nodes)
    assert dict(first.edges) == dict(second.edges)


def test_split_parallel_edges_shared_constants_and_multiple_results(tmp_path):
    x = ops.parameter([2, 4], ov.Type.f32, name="input")
    split = ops.split(x, ops.constant(np.int64(1)), 2, name="split")
    shared = ops.constant(np.array([1, 2], dtype=np.float32), name="shared")
    merged = ops.add(split.output(0), split.output(1), name="merged")
    left = ops.add(merged, shared, name="left")
    right = ops.add(split.output(1), shared, name="right")
    model = ov.Model([left, right], [x])
    graph = OpenVINOLoader().load(write_model(model, tmp_path))
    ids = {n.name: n.node_id for n in graph.nodes.values()}
    assert [(e.from_node, e.from_port, e.to_port) for e in graph.incoming(ids["merged"])] == [
        (ids["split"], 0, 0),
        (ids["split"], 1, 1),
    ]
    assert len(graph.outgoing(ids["split"])) == 3
    assert {(e.to_node, e.to_port) for e in graph.outgoing(ids["shared"])} == {
        (ids["left"], 1),
        (ids["right"], 1),
    }
    assert all(e.tensor.shape == TensorShape((2, 2)) for e in graph.outgoing(ids["split"]))
    assert sum(n.op_type == "Result" for n in graph.nodes.values()) == 2


@pytest.mark.parametrize(
    "partial_shape,expected",
    [
        (ov.PartialShape.dynamic(), TensorShape(None)),
        (ov.PartialShape([]), TensorShape(())),
        (ov.PartialShape([0, 3]), TensorShape((0, 3))),
        (ov.PartialShape([-1, 3]), TensorShape((None, 3))),
        (ov.PartialShape([ov.Dimension(1, 8), 3]), TensorShape((Dimension(1, 8), 3))),
        (ov.PartialShape([ov.Dimension(4, -1)]), TensorShape((Dimension(4),))),
    ],
)
def test_partial_shapes_preserve_rank_bounds_and_scalar(partial_shape, expected):
    x = ops.parameter(partial_shape, ov.Type.f32, name="input")
    graph = OpenVINOLoader().parse(ov.Model([x], [x]))
    edge = next(iter(graph.edges.values()))
    assert edge.tensor.shape == expected
    parameter = graph.nodes[edge.from_node]
    assert parameter.attributes["shape"] == str(expected)


def test_dynamic_dimension_bounds_survive_ir_loading(tmp_path):
    x = ops.parameter([ov.Dimension(1, 8), ov.Dimension(4, -1)], ov.Type.f32)
    graph = OpenVINOLoader().load(write_model(ov.Model([x], [x]), tmp_path))
    assert next(iter(graph.edges.values())).tensor.shape == TensorShape(
        (Dimension(1, 8), Dimension(4))
    )


@pytest.mark.parametrize(
    "element_type,expected",
    [(ov.Type.f16, "fp16"), (ov.Type.f64, "fp64"), (ov.Type.i64, "i64"), (ov.Type.dynamic, None)],
)
def test_element_types_and_unresolved_type(element_type, expected):
    x = ops.parameter([1], element_type)
    graph = OpenVINOLoader().parse(ov.Model([x], [x]))
    assert next(iter(graph.edges.values())).tensor.dtype == expected


def test_constants_are_never_read_as_values_or_attributes(linear_model, monkeypatch):
    original_attributes = ov.Node.get_attributes

    def attributes(node):
        assert node.get_type_name() != "Constant", "Constant values must not be requested"
        return original_attributes(node)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor values must not be accessed")

    monkeypatch.setattr(ov.Node, "get_attributes", attributes)
    monkeypatch.setattr(ov.op.Constant, "get_data", forbidden)
    monkeypatch.setattr(ov.op.Constant, "data", property(forbidden))
    graph = OpenVINOLoader().parse(linear_model)
    constants = [n for n in graph.nodes.values() if n.op_type == "Constant"]
    assert len(constants) == 2
    assert all(set(n.attributes) == {"dtype", "shape"} for n in constants)


def test_reading_does_not_compile_model(linear_model, tmp_path, monkeypatch):
    path = write_model(linear_model, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("The loader must not compile a model")

    monkeypatch.setattr(ov.Core, "compile_model", forbidden)
    assert len(OpenVINOLoader().load(path).nodes) == 6


def test_parse_and_graph_changes_leave_original_model_unchanged(linear_model, tmp_path):
    before = write_model(linear_model, tmp_path, "before")
    graph = OpenVINOLoader().parse(linear_model)
    for node in graph.nodes.values():
        node.attributes.clear()
        node.visualization["border_color"] = "red"
    graph.remove_node(next(iter(graph.nodes)))
    after = write_model(linear_model, tmp_path, "after")
    assert before.read_bytes() == after.read_bytes()
    assert before.with_suffix(".bin").read_bytes() == after.with_suffix(".bin").read_bytes()


def test_loader_and_pipeline_highlight_without_changing_model(linear_model, tmp_path):
    loader = OpenVINOLoader()
    model = loader.read_model(write_model(linear_model, tmp_path))
    before = write_model(model, tmp_path, "before_pipeline")
    graph = loader.parse(model)
    result = Pipeline(
        [
            HighlightPass(["matmul", "weights"]),
            PassStep(HighlightPass(["add"]), enabled=False),
        ]
    ).apply(graph)
    assert result is graph
    assert len(graph.nodes) == 6 and len(graph.edges) == 5
    highlighted = {
        node.name
        for node in graph.nodes.values()
        if node.visualization.get("border_color") == "red"
    }
    assert highlighted == {"matmul", "weights"}
    after = write_model(model, tmp_path, "after_pipeline")
    assert before.read_bytes() == after.read_bytes()
    assert before.with_suffix(".bin").read_bytes() == after.with_suffix(".bin").read_bytes()


def test_graph_contains_only_detached_data(linear_model):
    graph = OpenVINOLoader().parse(linear_model)

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                check(key)
                check(item)
        elif isinstance(value, tuple):
            for item in value:
                check(item)
        else:
            assert value is None or type(value) in (str, int, float, bool)

    for node in graph.nodes.values():
        check(node.attributes)
        check(node.visualization)


@pytest.mark.parametrize("reverse", [False, True])
def test_hiding_and_highlighting_preserve_original_model_and_weights(
    linear_model, tmp_path, reverse
):
    loader = OpenVINOLoader()
    model = loader.read_model(write_model(linear_model, tmp_path))
    before = write_model(model, tmp_path, "before_hiding")
    graph = loader.parse(model)
    constants = {
        node.name: node.node_id for node in graph.nodes.values() if node.op_type == "Constant"
    }
    steps = [HighlightPass(["matmul", "weights"]), HideConstantsPass()]
    Pipeline(reversed(steps) if reverse else steps).apply(graph)
    assert len(graph.nodes) == 4 and len(graph.edges) == 3
    nodes = {node.name: node for node in graph.nodes.values()}
    assert set(nodes) == {"input", "matmul", "add", "output"}
    assert nodes["matmul"].visualization["border_color"] == "red"
    weight = nodes["matmul"].visualization["hidden_constants"][1]
    bias = nodes["add"].visualization["hidden_constants"][1]
    assert weight.constant_id == constants["weights"]
    assert bias.constant_id == constants["bias"]
    assert str(weight) == "[1]: fp32[2, 1]"
    assert str(bias) == "[1]: fp32[1]"
    assert nodes["matmul"].attributes == {"transpose_a": False, "transpose_b": False}
    assert {
        (graph.nodes[e.from_node].name, e.from_port, graph.nodes[e.to_node].name, e.to_port)
        for e in graph.edges.values()
    } == {("input", 0, "matmul", 0), ("matmul", 0, "add", 0), ("add", 0, "output", 0)}
    after = write_model(model, tmp_path, "after_hiding")
    assert before.read_bytes() == after.read_bytes()
    assert before.with_suffix(".bin").read_bytes() == after.with_suffix(".bin").read_bytes()


def test_shared_constant_in_real_model_marks_all_consumer_ports(tmp_path):
    x = ops.parameter([2], ov.Type.f32, name="input")
    shared = ops.constant(np.array([1, 2], dtype=np.float32), name="weights")
    add = ops.add(x, shared, name="add")
    multiply = ops.multiply(shared, x, name="multiply")
    graph = OpenVINOLoader().load(write_model(ov.Model([add, multiply], [x]), tmp_path))
    constant_id = next(n.node_id for n in graph.nodes.values() if n.op_type == "Constant")
    Pipeline([HideConstantsPass()]).apply(graph)
    nodes = {node.name: node for node in graph.nodes.values()}
    for name, port in (("add", 1), ("multiply", 0)):
        record = nodes[name].visualization["hidden_constants"][port]
        assert record.constant_id == constant_id
        assert record.shared
        assert str(record) == f"[{port}]: fp32[2] (shared: {constant_id})"
    assert not any(n.op_type == "Constant" for n in graph.nodes.values())
    assert len(graph.edges) == 4


def test_vector_attributes_and_unused_parameter_ports_are_preserved():
    image = ops.parameter([1, 1, 5, 5], ov.Type.f32, name="image")
    unused = ops.parameter([ov.Dimension(2, 8)], ov.Type.f32, name="unused")
    filters = ops.constant(np.ones((1, 1, 3, 3), dtype=np.float32))
    convolution = ops.convolution(image, filters, [2, 2], [1, 1], [1, 1], [1, 1])
    model = ov.Model([convolution], [image, unused])
    graph = OpenVINOLoader().parse(model)
    node = next(n for n in graph.nodes.values() if n.op_type == "Convolution")
    assert node.attributes["strides"] == (2, 2)
    assert node.attributes["pads_begin"] == (1, 1)
    node.attributes["strides"] = (9, 9)
    assert list(convolution.get_attributes()["strides"]) == [2, 2]
    parameter = next(n for n in graph.nodes.values() if n.name == "unused")
    assert parameter.output_ports == (0,)
    assert parameter.attributes["shape"] == "[2..8]"
    assert graph.outgoing(parameter.node_id) == ()


def test_nested_if_keeps_outer_ports_and_documents_unexpanded_bodies():
    condition = ops.parameter([], ov.Type.boolean, name="condition")
    x = ops.parameter([2], ov.Type.f32, name="input")
    then_x = ops.parameter([2], ov.Type.f32)
    else_x = ops.parameter([2], ov.Type.f32)
    then_result = ops.result(ops.relu(then_x))
    else_result = ops.result(ops.negative(else_x))
    branch = ops.if_op(condition)
    branch.set_then_body(ov.Model([then_result], [then_x]))
    branch.set_else_body(ov.Model([else_result], [else_x]))
    branch.set_input(x.output(0), then_x, else_x)
    branch.set_output(then_result, else_result)
    graph = OpenVINOLoader().parse(ov.Model([branch], [condition, x]))
    node = next(n for n in graph.nodes.values() if n.op_type == "If")
    assert node.input_ports == (0, 1)
    assert node.output_ports == (0,)
    assert len(graph.incoming(node.node_id)) == 2
    assert len(graph.outgoing(node.node_id)) == 1
    assert "not expanded" in node.visualization["note"]
    assert not any(n.op_type in {"Relu", "Negative"} for n in graph.nodes.values())


def test_weightless_ir_can_load_without_bin(tmp_path):
    x = ops.parameter([1], ov.Type.f32)
    path = write_model(ov.Model([x], [x]), tmp_path)
    path.with_suffix(".bin").unlink()
    assert len(OpenVINOLoader().load(path).nodes) == 2


def test_explicit_weights_path_and_unicode_model_path(linear_model, tmp_path):
    path = write_model(linear_model, tmp_path / "model space \u03b1", "linear")
    weights = path.with_name("separate weights.bin")
    path.with_suffix(".bin").rename(weights)
    assert len(OpenVINOLoader().load(path, weights).nodes) == 6


def test_missing_required_weights_report_companion_path(linear_model, tmp_path):
    path = write_model(linear_model, tmp_path)
    path.with_suffix(".bin").unlink()
    with pytest.raises(ModelLoadError, match="Companion weights file was not found"):
        OpenVINOLoader().load(path)


def test_truncated_weights_report_model_error(linear_model, tmp_path):
    path = write_model(linear_model, tmp_path)
    path.with_suffix(".bin").write_bytes(b"\x00")
    with pytest.raises(ModelLoadError, match="Cannot read OpenVINO IR"):
        OpenVINOLoader().load(path)


@pytest.mark.parametrize("contents", ["not XML", "<net>", "<unrelated/>"])
def test_invalid_xml_has_meaningful_error(tmp_path, contents):
    path = tmp_path / "invalid.xml"
    path.write_text(contents)
    with pytest.raises(ModelLoadError, match="Cannot read OpenVINO IR"):
        OpenVINOLoader().load(path)


def test_missing_explicit_weights_are_rejected(linear_model, tmp_path):
    path = write_model(linear_model, tmp_path)
    with pytest.raises(ModelLoadError, match="Weights file does not exist"):
        OpenVINOLoader().load(path, tmp_path / "missing.bin")


@pytest.mark.parametrize("filename", ["missing.xml", "directory.xml"])
def test_missing_or_nonfile_model_is_rejected(tmp_path, filename):
    (tmp_path / "directory.xml").mkdir()
    with pytest.raises(ModelLoadError, match="Model file does not exist"):
        OpenVINOLoader().load(tmp_path / filename)


@pytest.mark.parametrize("filename", ["model.onnx", "model.bin", "model.txt"])
def test_other_model_formats_are_explicitly_rejected(tmp_path, filename):
    with pytest.raises(ModelLoadError, match="Expected OpenVINO IR"):
        OpenVINOLoader().load(tmp_path / filename)


def test_parse_rejects_non_model():
    with pytest.raises(ModelLoadError, match="Expected an openvino.Model"):
        OpenVINOLoader().parse(None)


def test_example_script_generates_loadable_model(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "create_example_models.py"
    subprocess.run([sys.executable, str(script), "--output-dir", str(tmp_path)], check=True)
    graph = OpenVINOLoader().load(tmp_path / "linear.xml")
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 5
    mlp = OpenVINOLoader().load(tmp_path / "mlp.xml")
    assert len(mlp.nodes) == 11
    assert len(mlp.edges) == 10
    assert sum(n.op_type == "MatMul" for n in mlp.nodes.values()) == 2
    assert sum(n.op_type == "Relu" for n in mlp.nodes.values()) == 1
