"""Measure real model loading, passes, and SVG rendering; no automatic downloads."""

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from model_graph_viewer.errors import RenderError
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline
from model_graph_viewer.renderers import GraphvizRenderer


def environment() -> dict:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "openvino": version("openvino"),
        "python_graphviz": version("graphviz"),
        "graphviz": subprocess.run(
            ["dot", "-V"], capture_output=True, text=True, check=True
        ).stderr.strip(),
    }
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_physical",
                    "available_physical",
                    "total_page",
                    "available_page",
                    "total_virtual",
                    "available_virtual",
                    "available_extended",
                )
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result["physical_memory_bytes"] = memory.total_physical
    return result


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": checksum}


def measure(
    path: Path,
    output_dir: Path,
    timeout: float,
    highlight: str | None = None,
    layout: str = "default",
) -> dict:
    report = {"xml": identity(path), "weights": identity(path.with_suffix(".bin")), "runs": []}
    for transformed in (False, True):
        variant = "after" if transformed else "before"
        print(f"{path}: {variant}: loading", flush=True)
        start = time.perf_counter()
        loader = OpenVINOLoader()
        model = loader.read_model(path)
        read_seconds = time.perf_counter() - start
        start = time.perf_counter()
        graph = loader.parse(model)
        construction_seconds = time.perf_counter() - start
        del model
        original = {"nodes": len(graph.nodes), "edges": len(graph.edges)}
        if highlight is None:
            highlight = next((n.name for n in graph.nodes.values() if n.op_type == "MatMul"), None)
            if highlight is None:
                highlight = next(
                    (
                        n.name
                        for n in graph.nodes.values()
                        if n.op_type not in {"Constant", "Parameter", "Result"}
                    ),
                    None,
                )
        steps = (
            [HighlightPass([highlight] if highlight is not None else []), HideConstantsPass()]
            if transformed
            else []
        )
        timings = []
        start = time.perf_counter()
        for step in steps:
            step_start = time.perf_counter()
            Pipeline([step]).apply(graph)
            timings.append(
                {"pass": type(step).__name__, "seconds": time.perf_counter() - step_start}
            )
        passes_seconds = time.perf_counter() - start
        print(
            f"  {original} -> {len(graph.nodes)} nodes, {len(graph.edges)} edges; rendering",
            flush=True,
        )
        filename = (
            f"{path.parent.name if path.stem == 'openvino_model' else path.stem}_{variant}.svg"
        )
        start = time.perf_counter()
        output = output_dir / filename
        error = None
        try:
            GraphvizRenderer(timeout=timeout, layout=layout).render(graph, output)
        except RenderError as exc:
            error = str(exc)
        rendering_seconds = time.perf_counter() - start
        report["runs"].append(
            {
                "variant": variant,
                "layout": layout,
                "status": "failed" if error else "success",
                "error": error,
                "before": original,
                "after": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
                "highlight_names": [highlight] if transformed and highlight is not None else [],
                "read_seconds": read_seconds,
                "construction_seconds": construction_seconds,
                "loading_seconds": read_seconds + construction_seconds,
                "passes_seconds": passes_seconds,
                "pass_timings": timings,
                "rendering_seconds": rendering_seconds,
                "svg": None if error else output.as_posix(),
                "svg_size_bytes": None if error else output.stat().st_size,
            }
        )
        total = read_seconds + construction_seconds + passes_seconds + rendering_seconds
        print(f"  {'FAILED: ' + error if error else 'completed'}; elapsed {total:.3f}s", flush=True)
        del graph
        gc.collect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/svg"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--layout", choices=("default", "fast"), default="default")
    args = parser.parse_args()
    report = {
        "environment": environment(),
        "method": (
            "One run per variant; baseline first, then transformed. "
            "Files are hashed before timing, "
            "so OS file caching is warm. No model compilation. "
            "Read and construction measured separately; pass times include validation; "
            "render includes DOT, native dot, SVG validation and file write. "
            "Times are measurements, not guarantees. Native peak memory is not measured."
        ),
        "models": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for path in args.model:
        report["models"].append(measure(path, args.output_dir, args.timeout, layout=args.layout))
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.report}", flush=True)
    return int(any(r["status"] != "success" for m in report["models"] for r in m["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
