"""Check native dependencies before attempting to visualize a large model."""

import platform
import shutil
import subprocess
from importlib import import_module
from importlib.metadata import version
from xml.etree import ElementTree

from model_graph_viewer.errors import EnvironmentCheckError


def check_environment() -> dict[str, str]:
    """Import the backends and render a tiny SVG entirely in memory."""
    dot = shutil.which("dot")
    if dot is None:
        raise EnvironmentCheckError(
            "Graphviz executable 'dot' was not found. Install Graphviz from "
            "https://graphviz.org/download/ and add its bin directory to PATH. "
            "The Python graphviz package alone does not install dot."
        )
    for name in ("openvino", "graphviz"):
        try:
            import_module(name)
        except (ImportError, OSError) as exc:
            raise EnvironmentCheckError(f"Cannot import {name}: {exc}") from exc

    import graphviz

    try:
        dot_version = subprocess.run(
            [dot, "-V"], capture_output=True, text=True, check=True, timeout=15
        )
        svg = graphviz.Source("digraph { input -> output }", format="svg").pipe()
    except (OSError, subprocess.SubprocessError, graphviz.ExecutableNotFound) as exc:
        raise EnvironmentCheckError(f"Graphviz could not render the test graph: {exc}") from exc
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        raise EnvironmentCheckError("Graphviz returned invalid SVG.") from exc
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise EnvironmentCheckError("Graphviz output is not an SVG document.")

    return {
        "Python": platform.python_version(),
        "OpenVINO": version("openvino"),
        "graphviz (Python)": version("graphviz"),
        "Graphviz (dot)": (dot_version.stderr or dot_version.stdout).strip(),
        "SVG smoke test": "passed",
    }
