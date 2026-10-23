"""Render before/after SVGs of the generated small OpenVINO models."""

import argparse
from pathlib import Path

from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline
from model_graph_viewer.renderers import GraphvizRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/svg"))
    args = parser.parse_args()
    renderer = GraphvizRenderer()
    for name, highlight in [("linear", "matmul"), ("branching", "split")]:
        graph = OpenVINOLoader().load(args.model_dir / f"{name}.xml")
        before = (len(graph.nodes), len(graph.edges))
        renderer.render(graph, args.output_dir / f"{name}_before.svg")
        Pipeline([HighlightPass([highlight]), HideConstantsPass()]).apply(graph)
        renderer.render(graph, args.output_dir / f"{name}_after.svg")
        print(f"{name}: {before} -> {(len(graph.nodes), len(graph.edges))} (nodes, edges)")


if __name__ == "__main__":
    main()
