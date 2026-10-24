"""Public operations and a convenience model-to-SVG pipeline.

Backends are imported only when used. Graph and pass operations remain independent
of OpenVINO and Graphviz, including when imported from the package root.
"""

from collections.abc import Iterable
from pathlib import Path

from model_graph_viewer.errors import EnvironmentCheckError
from model_graph_viewer.graph import Graph
from model_graph_viewer.passes import PassStep, Pipeline, TransformationPass


def load_graph(model_path: str | Path, *, weights_path: str | Path | None = None) -> Graph:
    """Read OpenVINO IR into a detached graph without compiling the model."""
    try:
        from model_graph_viewer.loaders.openvino import OpenVINOLoader
    except (ImportError, OSError) as exc:
        raise EnvironmentCheckError(f"Cannot import the OpenVINO backend: {exc}") from exc
    return OpenVINOLoader().load(model_path, weights_path=weights_path)


def apply_passes(graph: Graph, passes: Iterable[TransformationPass | PassStep] = ()) -> Graph:
    """Apply enabled passes in order, returning the same graph; no rollback."""
    return Pipeline(passes).apply(graph)


def render_graph(
    graph: Graph,
    output_path: str | Path,
    *,
    max_label_length: int = 120,
    timeout: float | None = None,
) -> Path:
    """Write SVG atomically and return its path; do not change the graph."""
    try:
        from model_graph_viewer.renderers import GraphvizRenderer
    except (ImportError, OSError) as exc:
        raise EnvironmentCheckError(f"Cannot import the Graphviz backend: {exc}") from exc
    return GraphvizRenderer(max_label_length=max_label_length, timeout=timeout).render(
        graph, output_path
    )


def visualize(
    model_path: str | Path,
    output_path: str | Path,
    *,
    passes: Iterable[TransformationPass | PassStep] = (),
    weights_path: str | Path | None = None,
    max_label_length: int = 120,
    timeout: float | None = None,
) -> Path:
    """Load IR, run the supplied passes, write SVG, and return its path.

    By default no transformation is applied. PassStep supports disabling a step;
    repeated passes execute repeatedly. Original model files remain untouched.
    Use the individual operations when the resulting Graph is also needed.
    """
    pipeline = Pipeline(passes)
    graph = load_graph(model_path, weights_path=weights_path)
    pipeline.apply(graph)
    return render_graph(graph, output_path, max_label_length=max_label_length, timeout=timeout)
