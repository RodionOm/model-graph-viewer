"""Exercise the public API and CLI against actual IR files and native Graphviz."""

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import apply_passes, load_graph, render_graph, visualize
from model_graph_viewer.errors import ModelLoadError
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, PassStep

NS = {"s": "http://www.w3.org/2000/svg"}
NAME = "узел, с пробелом"


@pytest.fixture
def model_path(tmp_path):
    directory = tmp_path / "модель with spaces"
    directory.mkdir()
    x = ops.parameter([1, 2], ov.Type.f32, name="input")
    w = ops.constant(np.array([[0.5], [1]], dtype=np.float32), name="weights")
    b = ops.constant(np.array([1], dtype=np.float32), name="bias")
    product = ops.matmul(x, w, False, False, name=NAME)
    result = ops.result(ops.add(product, b, name=NAME), name="output")
    path = directory / "линейная model.xml"
    ov.serialize(ov.Model([result], [x]), path)
    return path


def run_cli(*args, **env_changes):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_changes}
    return subprocess.run(
        [sys.executable, "-m", "model_graph_viewer", *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


def semantic_svg(path):
    root = ET.parse(path).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    # Compare content and styling, without depending on exact layout coordinates.
    return sorted(
        (
            group.get("class"),
            group.find("s:title", NS).text,
            tuple(t.text for t in group.findall(".//s:text", NS)),
            tuple((p.get("stroke"), p.get("fill")) for p in group.findall(".//s:polygon", NS)),
        )
        for group in root.findall(".//s:g", NS)
        if group.get("class") in {"node", "edge"}
    )


def test_default_api_and_cli_render_untransformed_graph(model_path, tmp_path):
    api_path = visualize(model_path, tmp_path / "api.svg")
    cli_path = tmp_path / "cli.svg"
    result = run_cli("--model", model_path, "--output", cli_path)
    assert result.returncode == 0, result.stderr
    assert "Created SVG" in result.stdout
    expected = semantic_svg(api_path)
    assert semantic_svg(cli_path) == expected
    assert sum(item[0] == "node" for item in expected) == 6
    assert sum(item[0] == "edge" for item in expected) == 5
    assert not any(stroke == "red" for item in expected for stroke, _ in item[3])


@pytest.mark.parametrize(
    "pass_names",
    [
        "highlight,hide-constants",
        "hide-constants,highlight",
        "highlight,hide-constants,hide-constants,highlight",
    ],
)
def test_api_cli_equivalence_unicode_names_repeated_passes_and_custom_weights(
    model_path, tmp_path, pass_names
):
    weights = model_path.with_name("отдельные weights.bin")
    model_path.with_suffix(".bin").rename(weights)
    original = (model_path.read_bytes(), weights.read_bytes())
    passes = [
        HighlightPass([NAME]) if name == "highlight" else HideConstantsPass()
        for name in pass_names.split(",")
    ]
    api_path = visualize(
        model_path,
        tmp_path / "API граф.svg",
        passes=iter(passes),
        weights_path=weights,
        timeout=30,
        max_label_length=100,
    )
    cli_path = tmp_path / "вложенный каталог" / "CLI граф.svg"
    result = run_cli(
        "--model",
        model_path,
        "--weights",
        weights,
        "--output",
        cli_path,
        "--passes",
        pass_names,
        "--highlight-node",
        NAME,
        "--timeout",
        "30",
        "--max-label-length",
        "100",
    )
    assert result.returncode == 0, result.stderr
    assert str(cli_path) in result.stdout
    expected = semantic_svg(api_path)
    assert semantic_svg(cli_path) == expected
    assert sum(item[0] == "node" for item in expected) == 4
    assert sum(item[0] == "edge" for item in expected) == 3
    highlighted = [item for item in expected if any(s == "red" for s, _ in item[3])]
    assert len(highlighted) == 2  # Both operations have exactly the same friendly name.
    assert all(NAME in item[2] for item in highlighted)
    assert (model_path.read_bytes(), weights.read_bytes()) == original


def test_separate_api_operations_and_disabled_pass_match_convenience_call(model_path, tmp_path):
    steps = [PassStep(HideConstantsPass(), enabled=False), HighlightPass(["input"])]
    graph = load_graph(model_path)
    assert apply_passes(graph, steps) is graph
    assert len(graph.nodes) == 6
    separate = render_graph(graph, tmp_path / "separate.svg")
    together = visualize(model_path, tmp_path / "together.svg", passes=steps)
    assert semantic_svg(separate) == semantic_svg(together)


def test_cli_missing_dot_preserves_output(model_path, tmp_path):
    output = tmp_path / "out.svg"
    output.write_bytes(b"previous SVG")
    result = run_cli("--model", model_path, "--output", output, PATH="")
    assert result.returncode == 1
    assert "Graphviz executable 'dot' was not found" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.read_bytes() == b"previous SVG"


@pytest.mark.parametrize("case", ["missing-model", "invalid-xml", "missing-weights", "bad-output"])
def test_cli_reports_file_errors_without_success_or_traceback(model_path, tmp_path, case):
    output = tmp_path / "out.svg"
    args = []
    if case == "missing-model":
        model_path = tmp_path / "missing.xml"
    elif case == "invalid-xml":
        model_path.write_text("not an IR")
    elif case == "missing-weights":
        args = ["--weights", tmp_path / "missing.bin"]
    else:
        # A regular file cannot be the parent directory of an SVG.
        parent = tmp_path / "file"
        parent.write_text("keep")
        output = parent / "out.svg"
    result = run_cli("--model", model_path, "--output", output, *args)
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout
    assert not output.exists()


def test_api_retains_contextual_errors(model_path, tmp_path):
    model_path.write_text("invalid XML")
    with pytest.raises(ModelLoadError) as exc:
        visualize(model_path, tmp_path / "out.svg")
    assert str(model_path) in str(exc.value)
    assert exc.value.__cause__ is not None


def test_installed_console_command_uses_same_pipeline(model_path, tmp_path):
    executable = Path(sys.executable).parent / (
        "model-graph-viewer.exe" if os.name == "nt" else "model-graph-viewer"
    )
    output = tmp_path / "console.svg"
    result = subprocess.run(
        [
            str(executable),
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--passes",
            "hide-constants",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sum(item[0] == "node" for item in semantic_svg(output)) == 4
