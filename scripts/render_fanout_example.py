"""Render a small real OpenVINO graph demonstrating clickable fan-out references."""

import argparse
from pathlib import Path

import openvino as ov
from openvino import opset13 as ops

from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import FanoutReferencesPass, HighlightPass, Pipeline
from model_graph_viewer.renderers import GraphvizRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/svg"))
    args = parser.parse_args()
    source = ops.parameter([1, 4], ov.Type.f32, name="shared_input_for_four_processing_branches")
    model = ov.Model(
        [
            ops.result(ops.relu(source, name=f"processing_branch_{i}"), name=f"result_{i}")
            for i in range(4)
        ],
        [source],
        "fanout_example",
    )
    graph = OpenVINOLoader().parse(model)
    HighlightPass(["processing_branch_0"]).apply(graph)
    GraphvizRenderer().render(graph, args.output_dir / "fanout_before.svg")
    Pipeline([FanoutReferencesPass()]).apply(graph)
    GraphvizRenderer(compact=True).render(graph, args.output_dir / "fanout_after.svg")
    print(f"Created fanout_before.svg, fanout_after.svg and fanout_after.json in {args.output_dir}")


if __name__ == "__main__":
    main()
