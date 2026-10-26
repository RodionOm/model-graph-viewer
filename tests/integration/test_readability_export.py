"""Exercise OpenVINO, CLI, real Graphviz links and JSON connectivity together."""

import hashlib
import json
from xml.etree import ElementTree

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from model_graph_viewer import cli, load_graph


def test_compact_fanout_svg_and_sidecar_preserve_real_model(tmp_path):
    x = ops.parameter([1, 2], ov.Type.f32, name="long input name " * 20)
    weight = ops.constant(np.array([1, 2], dtype=np.float32), name="weight")
    branches = [ops.add(x, weight, name=f"branch {i}") for i in range(4)]
    model = ov.Model([ops.result(b) for b in branches], [x])
    path = tmp_path / "model.xml"
    ov.serialize(model, path)
    original = {p: p.read_bytes() for p in (path, path.with_suffix(".bin"))}
    graph = load_graph(path)
    output = tmp_path / "readable.svg"
    assert (
        cli.main(
            [
                "--model",
                str(path),
                "--output",
                str(output),
                "--compact",
                "--passes",
                "highlight,hide-constants,fanout-references",
                "--highlight-node",
                "branch 0",
                "--timeout",
                "30",
            ]
        )
        == 0
    )
    details = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    svg = ElementTree.fromstring(output.read_bytes())
    ns = {"s": "http://www.w3.org/2000/svg"}
    text = " ".join(t.text or "" for t in svg.findall(".//s:text", ns))
    assert "long input name" not in text
    assert "branch 0" not in text
    assert "in 0 ← ID:" in text
    assert 'stroke="red"' in output.read_text(encoding="utf-8")
    svg_nodes = {g.attrib["id"] for g in svg.findall(".//s:g[@class='node']", ns)}
    assert svg_nodes == {n["svg_id"] for n in details["nodes"].values()}
    links = svg.findall(".//s:a", ns)
    targets = [a.attrib.get("{http://www.w3.org/1999/xlink}href", "") for a in links]
    assert sum(t.startswith("#node_") for t in targets) == 4
    assert all(t[1:] in svg_nodes for t in targets if t.startswith("#"))
    drawn = svg.findall(".//s:g[@class='edge']", ns)
    assert len(drawn) == sum(e["rendered"] for e in details["edges"]) == 4
    assert len(details["edges"]) == 8
    # Reconstruct the complete original port connectivity, including hidden constants.
    recovered = {
        (e["from_node"], e["from_port"], e["to_node"], e["to_port"]) for e in details["edges"]
    }
    for node_id, node in details["nodes"].items():
        for record in node["hidden_constants"]:
            recovered.add(
                (record["constant_id"], record["output_port"], node_id, record["input_port"])
            )
    assert recovered == {
        (e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges.values()
    }
    assert details["svg_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert all(p.read_bytes() == data for p, data in original.items())
