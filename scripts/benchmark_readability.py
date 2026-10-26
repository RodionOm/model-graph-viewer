"""Reproduce the fan-out investigation and TinyLlama readability comparison."""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
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
    boxes = []
    transform = root.find("s:g", NS).attrib["transform"]
    tx, ty = map(float, re.search(r"translate\(([^)]+)\)", transform).group(1).split())
    for group in root.findall(".//s:g[@class='node']", NS):
        polygon = group.find(".//s:polygon", NS)
        points = [tuple(map(float, point.split(","))) for point in polygon.attrib["points"].split()]
        widths.append(max(x for x, _ in points) - min(x for x, _ in points))
        boxes.append(
            (
                min(x for x, _ in points) + tx,
                max(x for x, _ in points) + tx,
                min(y for _, y in points) + ty,
                max(y for _, y in points) + ty,
            )
        )
    viewbox = list(map(float, root.attrib["viewBox"].split()))
    bands = []
    for index in range(10):
        top = viewbox[1] + index * viewbox[3] / 10
        bottom = top + viewbox[3] / 10
        members = [box for box in boxes if box[2] < bottom and box[3] >= top]
        left = min((box[0] for box in members), default=0)
        right = max((box[1] for box in members), default=0)
        bands.append(
            {
                "band": index + 1,
                "y_min": top,
                "y_max": bottom,
                "node_count": len(members),
                "x_min": left,
                "x_max": right,
                "width": right - left,
                "fraction_of_canvas": (right - left) / viewbox[2],
            }
        )
    return {
        "node_count": len(widths),
        "drawn_edge_count": len(root.findall(".//s:g[@class='edge']", NS)),
        "reference_link_count": sum(
            link.attrib.get("{http://www.w3.org/1999/xlink}href", "").startswith("#node_")
            for link in root.findall(".//s:a", NS)
        ),
        "viewbox": viewbox,
        "horizontal_bands": bands,
        "median_node_width_pt": statistics.median(widths),
        "max_node_width_pt": max(widths),
        "svg_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def drift_experiment(graph, report: dict, directory: Path, timeout: float) -> dict:
    """Compare DOT hints on identical labels, without changing the model graph."""
    Pipeline([FanoutReferencesPass()]).apply(graph)
    dot = GraphvizRenderer(compact=True).to_dot(graph)
    edges = [
        (i, e)
        for i, e in enumerate(graph.edges.values())
        if graph.nodes[e.from_node].op_type == graph.nodes[e.to_node].op_type == "Add"
        and not graph.nodes[e.from_node].visualization.get("reference_fanout", False)
    ]
    # Direct Add->Add edges form the residual backbone in this pinned model.
    # Restrict the 22-edge trial to edges within one decoder layer.
    layer = re.compile(r"model\.layers\.(\d+)/")
    within = set()
    for i, edge in edges:
        first = layer.search(graph.nodes[edge.from_node].name)
        second = layer.search(graph.nodes[edge.to_node].name)
        if first and second and first[1] == second[1]:
            within.add(i)
    all_edges = {i for i, _ in edges}
    names = {node_id: f"n{index}" for index, node_id in enumerate(graph.nodes)}
    chain = {node_id for _, edge in edges for node_id in (edge.from_node, edge.to_node)}
    result = {
        "labels": "Current renderer labels, unchanged between the five trials.",
        "method": "Same graph, labels and default dot layout for all trials; only weight/group "
        "hints differ. Ten equal-height canvas bands, top to bottom. Each width spans all node "
        "boxes intersecting the band (not edge routes or empty margins). Bands are intervals, "
        "not single-height slices. Coordinates in SVG points, with graph translation applied.",
        "environment": report["environment"],
        "within_layer_add_edges": len(within),
        "all_add_to_add_edges": len(edges),
        "backbone_nodes": len(chain),
        "trials": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    for variant, weighted, grouped in [
        ("baseline", set(), False),
        ("weight10_within_layer", within, False),
        ("weight10_backbone", all_edges, False),
        ("group_backbone", set(), True),
        ("weight10_and_group", all_edges, True),
    ]:
        trial = dot.copy()
        # Remove the retained production hint, if present, to reproduce the control.
        trial.body = [
            re.sub(r" weight=10\b| group=residual_backbone\b", "", line) for line in trial.body
        ]
        for index, line in enumerate(trial.body):
            match = re.search(r"\bid=edge_(\d+)\b", line)
            if match and int(match[1]) in weighted:
                before, after = line.rsplit("]", 1)
                trial.body[index] = before + " weight=10]" + after
        if grouped:
            for node_id in graph.nodes:
                if node_id in chain:
                    trial.node(names[node_id], group="residual_backbone")
        print(f"Drift trial: {variant}", flush=True)
        start = time.perf_counter()
        process = subprocess.run(
            ["dot", "-Tsvg"],
            input=trial.source.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output = directory / f"{variant}.svg"
        output.write_bytes(process.stdout)
        metrics = svg_metrics(output)
        assert metrics["node_count"] == len(graph.nodes)
        result["trials"].append(
            {
                "variant": variant,
                "seconds": time.perf_counter() - start,
                "output": output.as_posix(),
                **metrics,
            }
        )
        print(
            f"  canvas: {metrics['viewbox'][2:]}; bands: "
            f"{[round(b['width'], 2) for b in metrics['horizontal_bands']]}",
            flush=True,
        )
    return result


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
    parser.add_argument(
        "--drift-experiment",
        action="store_true",
        help="compare residual weight/group hints on identical compact labels",
    )
    args = parser.parse_args()
    existing = json.loads(args.report.read_text(encoding="utf-8")) if args.report.exists() else {}
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
        ("compact_backbone", True, False, "fast", args.baseline_dir / "compact_backbone.svg"),
        ("compact_references", True, True, "default", args.output),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.drift_experiment:
        existing["diagonal_drift"] = drift_experiment(
            graph, report, args.baseline_dir / "drift", args.timeout
        )
        args.report.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return
    if "diagonal_drift" in existing:
        report["diagonal_drift"] = existing["diagonal_drift"]
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
