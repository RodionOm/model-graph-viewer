import json
import runpy
import sys
from pathlib import Path

import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer.errors import RenderError
from model_graph_viewer.renderers import GraphvizRenderer


def test_benchmark_reports_failed_layouts_without_claiming_svg_success(tmp_path, monkeypatch):
    x = ops.parameter([1], ov.Type.f32, name="input")
    model = tmp_path / "model.xml"
    ov.serialize(ov.Model([ops.relu(x)], [x]), model)
    report = tmp_path / "report.json"

    def fail(*args, **kwargs):
        raise RenderError("Graphviz exceeded the 1-second timeout.")

    monkeypatch.setattr(GraphvizRenderer, "render", fail)
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--model",
            str(model),
            "--report",
            str(report),
            "--output-dir",
            str(tmp_path / "svg"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 1
    runs = json.loads(report.read_text())["models"][0]["runs"]
    assert len(runs) == 2
    for result in runs:
        assert result["status"] == "failed"
        assert "timeout" in result["error"]
        assert result["svg"] is None
        assert result["svg_size_bytes"] is None
        assert result["rendering_seconds"] >= 0


def test_band_measurements_use_node_boxes_and_graph_translation(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    metrics = runpy.run_path(str(scripts / "benchmark_readability.py"))["svg_metrics"]
    svg = tmp_path / "bands.svg"
    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <g transform="scale(1 1) rotate(0) translate(10 100)">
      <g class="node"><polygon points="0,-100 20,-100 20,-91 0,-91"/></g>
      <g class="node"><polygon points="50,-95 60,-95 60,-85 50,-85"/></g>
      <g class="edge"><path d="M0,0L999,999"/></g>
    </g></svg>""",
        encoding="utf-8",
    )
    bands = metrics(svg)["horizontal_bands"]
    assert len(bands) == 10
    assert (bands[0]["x_min"], bands[0]["x_max"], bands[0]["width"]) == (10, 70, 60)
    assert bands[0]["node_count"] == 2
    assert bands[0]["fraction_of_canvas"] == 0.6
    assert bands[1]["width"] == 10  # Second box intersects both first and second bands.
    assert all(b["width"] == b["node_count"] == 0 for b in bands[2:])
