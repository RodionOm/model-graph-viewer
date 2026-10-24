import json
import os
import subprocess
import sys
from xml.etree import ElementTree as ET

import openvino as ov
from openvino import opset13 as ops

from model_graph_viewer import load_graph
from model_graph_viewer.errors import ModelLoadError
from model_graph_viewer.loaders.openvino import OpenVINOLoader


def test_new_format_and_configured_pass_work_in_cli_without_core_edits(tmp_path):
    (tmp_path / "local_extension.py").write_text("""
import json
from pathlib import Path
from model_graph_viewer import Graph, Node

class Loader:
    def load(self, path):
        return Graph(Node(**node) for node in json.loads(Path(path).read_text()))

class Color:
    def __init__(self, color):
        self.color = color
    def apply(self, graph):
        for node in graph.nodes.values():
            node.visualization['border_color'] = self.color
        return graph
""")
    model = tmp_path / "model.custom"
    model.write_text(json.dumps([{"node_id": "x", "name": "custom node", "op_type": "Unknown"}]))
    config = tmp_path / "passes.json"
    config.write_text(
        json.dumps(
            [
                {"name": "local_extension:Color", "options": {"color": "blue"}},
                {"name": "local_extension:Color", "options": {"color": "green"}, "enabled": False},
                {"name": "highlight", "options": {"node_names": ["custom node"]}},
            ]
        )
    )
    output = tmp_path / "output.svg"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_graph_viewer",
            "--model",
            str(model),
            "--output",
            str(output),
            "--loader",
            "local_extension:Loader",
            "--pass-config",
            str(config),
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    root = ET.parse(output).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert root.find(".//s:polygon[@stroke='red']", ns) is not None
    assert root.find(".//s:polygon[@stroke='green']", ns) is None


def test_parameter_convert_and_variable_metadata_use_consistent_notation():
    x = ops.parameter([-1, -1, 8], ov.Type.f32, name="input")
    read = ops.read_value(x, "cache", ov.Type.f32, ov.PartialShape([-1, -1, 8]), name="read")
    converted = ops.convert(read, ov.Type.f16, name="convert")
    graph = OpenVINOLoader().parse(ov.Model([converted], [x]))
    nodes = {n.name: n for n in graph.nodes.values()}
    assert nodes["input"].attributes["element_type"] == "fp32"
    assert nodes["read"].attributes["variable_type"] == "fp32"
    assert nodes["read"].attributes["variable_shape"] == "[?, ?, 8]"
    assert nodes["convert"].attributes["destination_type"] == "fp16"


def test_broken_xml_does_not_incorrectly_blame_missing_weights(tmp_path):
    import pytest

    path = tmp_path / "bad.xml"
    path.write_text("not xml")
    with pytest.raises(ModelLoadError) as exc:
        load_graph(path)
    assert "Companion weights" not in str(exc.value)
