"""Measure graph/pass scaling; generate a separate 5002-node OpenVINO stress fixture."""

import argparse
import json
import statistics
import time
from pathlib import Path

import openvino as ov
from openvino import opset13 as ops

from model_graph_viewer import Edge, Graph, Node
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/scaling.json"))
    args = parser.parse_args()
    rows = []
    for size in (1000, 5000, 10000):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            graph = Graph(
                [
                    Node("input", "input", "Parameter", output_ports=(0,)),
                    Node("weight", "weight", "Constant", output_ports=(0,)),
                ]
            )
            previous = "input"
            for i in range(size):
                node_id = str(i)
                graph.add_node(Node(node_id, node_id, "Add", input_ports=(0, 1), output_ports=(0,)))
                graph.add_edge(Edge(previous, 0, node_id, 0))
                graph.add_edge(Edge("weight", 0, node_id, 1))
                previous = node_id
            construction = time.perf_counter() - start
            start = time.perf_counter()
            Pipeline([HighlightPass(["0"]), HideConstantsPass()]).apply(graph)
            passes = time.perf_counter() - start
            samples.append({"construction_seconds": construction, "passes_seconds": passes})
        rows.append(
            {
                "before_nodes": size + 2,
                "before_edges": 2 * size,
                "after_nodes": len(graph.nodes),
                "after_edges": len(graph.edges),
                "median_construction_seconds": statistics.median(
                    s["construction_seconds"] for s in samples
                ),
                "median_passes_seconds": statistics.median(s["passes_seconds"] for s in samples),
                "samples": samples,
            }
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "method": "3 runs per size; median; Add chain with shared Constant; no rendering.",
                "sizes": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    x = ops.parameter([1, 8], ov.Type.f32, name="input")
    value = x
    for i in range(5000):
        value = ops.relu(value, name=f"relu_{i}")
    model = ov.Model([ops.result(value, name="output")], [x], "synthetic_chain_5002")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ov.serialize(model, args.output_dir / "synthetic_5002.xml")
    print(f"Saved {args.report} and {args.output_dir / 'synthetic_5002.xml'}")


if __name__ == "__main__":
    main()
