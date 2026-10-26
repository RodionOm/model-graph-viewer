"""Reproduce the fan-out investigation and TinyLlama readability comparison."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from xml.etree import ElementTree

from benchmark import environment, identity

from model_graph_viewer import load_graph
from model_graph_viewer.passes import (
    FanoutReferencesPass,
    HideConstantsPass,
    HighlightPass,
    Pipeline,
)
from model_graph_viewer.renderers import GraphvizRenderer

NS = {"s": "http://www.w3.org/2000/svg"}
HIGHLIGHT = "__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul"


def svg_metrics(path: Path) -> dict:
    root = ElementTree.parse(path).getroot()
    widths = []
    for group in root.findall(".//s:g[@class='node']", NS):
        polygon = group.find(".//s:polygon", NS)
        points = [tuple(map(float, point.split(","))) for point in polygon.attrib["points"].split()]
        widths.append(max(x for x, _ in points) - min(x for x, _ in points))
    return {
        "node_count": len(widths),
        "drawn_edge_count": len(root.findall(".//s:g[@class='edge']", NS)),
        "reference_link_count": sum(
            link.attrib.get("{http://www.w3.org/1999/xlink}href", "").startswith("#node_")
            for link in root.findall(".//s:a", NS)
        ),
        "viewbox": list(map(float, root.attrib["viewBox"].split())),
        "median_node_width_pt": statistics.median(widths),
        "max_node_width_pt": max(widths),
        "svg_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_connectivity(graph, details: dict, original: set) -> None:
    edges = details["edges"]
    restored = {(e["from_node"], e["from_port"], e["to_node"], e["to_port"]) for e in edges}
    assert restored == {
        (e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges.values()
    }
    for node_id, node in details["nodes"].items():
        for record in node["hidden_constants"]:
            restored.add(
                (record["constant_id"], record["output_port"], node_id, record["input_port"])
            )
    assert restored == original, "Export lost original model connectivity"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/tinyllama/openvino_model.xml"))
    parser.add_argument("--output", type=Path, default=Path("examples/svg/tinyllama_after.svg"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/readability.json"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/readability"))
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = {
        "environment": environment(),
        "model": identity(args.model),
        "weights": identity(args.model.with_suffix(".bin")),
        "method": "One sequential run per variant, warm files after hashing; same loaded graph. "
        "Render includes DOT, layout, SVG and (when compact) JSON export. "
        "Timing is not a guarantee; canvas size is not a readability score. "
        "Previous layout uses fast; final layout uses default spline routing.",
        "variants": [],
    }
    start = time.perf_counter()
    graph = load_graph(args.model)
    report["load_seconds"] = time.perf_counter() - start
    original = {(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges.values()}
    report["original_connections"] = len(original)
    Pipeline([HighlightPass([HIGHLIGHT]), HideConstantsPass()]).apply(graph)
    high_fanout = sorted(
        (n for n in graph.nodes.values() if len(graph.outgoing(n.node_id)) > 3),
        key=lambda n: (-len(graph.outgoing(n.node_id)), n.node_id),
    )
    report["diagnosis"] = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "names_over_60_characters": sum(len(n.name) > 60 for n in graph.nodes.values()),
        "high_fanout_sources": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "op_type": n.op_type,
                "outgoing_edges": len(graph.outgoing(n.node_id)),
            }
            for n in high_fanout
        ],
    }
    variants = [
        ("previous", False, False, "fast", args.baseline_dir / "previous.svg"),
        ("compact_only", True, False, "fast", args.baseline_dir / "compact_only.svg"),
        ("compact_references", True, True, "default", args.output),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, compact, references, layout, output in variants:
        print(f"Rendering {name}", flush=True)
        pass_seconds = 0
        if references:
            start = time.perf_counter()
            Pipeline([FanoutReferencesPass(threshold=3)]).apply(graph)
            pass_seconds = time.perf_counter() - start
        renderer = GraphvizRenderer(compact=compact, layout=layout, timeout=args.timeout)
        start = time.perf_counter()
        renderer.render(graph, output)
        elapsed = time.perf_counter() - start
        metrics = svg_metrics(output)
        details = (
            json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            if compact
            else renderer.metadata(graph)
        )
        check_connectivity(graph, details, original)
        assert metrics["node_count"] == len(graph.nodes)
        assert metrics["drawn_edge_count"] == sum(e["rendered"] for e in details["edges"])
        assert metrics["reference_link_count"] == sum(not e["rendered"] for e in details["edges"])
        report["variants"].append(
            {
                "name": name,
                "layout": layout,
                "compact": compact,
                "render_seconds": elapsed,
                "reference_pass_seconds": pass_seconds,
                "all_original_connections_recovered": True,
                "graph_edges": len(graph.edges),
                "output": output.as_posix(),
                **metrics,
            }
        )
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  {elapsed:.3f}s; {metrics}", flush=True)
    assert identity(args.model) == report["model"]
    assert identity(args.model.with_suffix(".bin")) == report["weights"]
    report["model_files_unchanged"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
