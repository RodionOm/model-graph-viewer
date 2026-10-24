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
