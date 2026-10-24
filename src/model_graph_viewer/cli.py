"""Validate command-line options and invoke the public visualization API."""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from model_graph_viewer import __version__
from model_graph_viewer.api import visualize
from model_graph_viewer.configuration import component, configured_passes
from model_graph_viewer.environment import check_environment
from model_graph_viewer.errors import ViewerError
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, TransformationPass


def _timeout(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number of seconds")
    return number


def _label_length(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max label length must be an integer of at least 8"
        ) from exc
    if number < 8:
        raise argparse.ArgumentTypeError("max label length must be an integer of at least 8")
    return number


def _passes(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[TransformationPass]:
    names = [name.strip() for group in (args.passes or []) for name in group.split(",")]
    for name in names:
        if name not in {"highlight", "hide-constants"}:
            parser.error(f"unknown pass {name!r}; choose highlight or hide-constants")
    if args.highlight_node is not None and "highlight" not in names:
        parser.error("--highlight-node requires highlight in --passes")
    if "highlight" in names and args.highlight_node is None:
        parser.error("the highlight pass requires at least one --highlight-node")
    return [
        HighlightPass(args.highlight_node) if name == "highlight" else HideConstantsPass()
        for name in names
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-graph-viewer",
        description="Render a model as a port-aware SVG graph (OpenVINO IR by default).",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="check OpenVINO and Graphviz, including a small SVG render",
    )
    parser.add_argument("--model", type=Path, help="input model path (default: OpenVINO IR .xml)")
    parser.add_argument("--loader", metavar="MODULE:FACTORY", help="custom model loader factory")
    parser.add_argument(
        "--layout",
        choices=("default", "fast"),
        help="Graphviz layout profile (default: default); fast bounds layout optimization",
    )
    parser.add_argument(
        "--pass-config",
        type=Path,
        metavar="JSON",
        help="JSON array of independently configured passes; replaces --passes/--highlight-node",
    )
    parser.add_argument(
        "--weights", type=Path, help="optional .bin weights path (default: beside XML)"
    )
    parser.add_argument(
        "--output", type=Path, help="destination .svg path; creates parent directories"
    )
    parser.add_argument(
        "--passes",
        action="append",
        metavar="PASS[,PASS...]",
        help="ordered highlight,hide-constants passes; repeatable; default: no passes",
    )
    parser.add_argument(
        "--highlight-node",
        action="append",
        metavar="NAME",
        help="exact node name; repeat for more names; requires the highlight pass",
    )
    parser.add_argument(
        "--max-label-length",
        type=_label_length,
        metavar="N",
        help="maximum ordinary node field length, at least 8 (default: 120)",
    )
    parser.add_argument(
        "--timeout",
        type=_timeout,
        metavar="SECONDS",
        help="Graphviz layout time limit (default: no limit)",
    )
    args = parser.parse_args(argv)
    render_options = (
        args.model,
        args.weights,
        args.output,
        args.passes,
        args.highlight_node,
        args.max_label_length,
        args.timeout,
        args.loader,
        args.pass_config,
        args.layout,
    )
    has_render_options = any(value is not None for value in render_options)
    if args.check_environment and has_render_options:
        parser.error("--check-environment cannot be combined with model rendering options")
    if not args.check_environment and not has_render_options:
        parser.print_help()
        return 0
    transformations = []
    if not args.check_environment:
        if args.model is None or args.output is None:
            parser.error("rendering requires both --model and --output")
        if args.output.suffix.lower() != ".svg":
            parser.error("--output must end with .svg")
        if args.loader is not None and args.weights is not None:
            parser.error(
                "--weights belongs to the default loader; configure custom weights in its factory"
            )
        if args.pass_config is not None:
            if args.passes is not None or args.highlight_node is not None:
                parser.error("--pass-config cannot be combined with --passes or --highlight-node")
        else:
            transformations = _passes(parser, args)
    try:
        if args.check_environment:
            for name, value in check_environment().items():
                print(f"{name}: {value}")
        else:
            if args.pass_config is not None:
                transformations = configured_passes(args.pass_config)
            loader = component(args.loader) if args.loader is not None else None
            output = visualize(
                args.model,
                args.output,
                passes=transformations,
                weights_path=args.weights,
                loader=loader,
                max_label_length=args.max_label_length or 120,
                timeout=args.timeout,
                layout=args.layout or "default",
            )
            print(f"Created SVG: {output}")
    except ViewerError as exc:
        # Backend diagnostics can span many lines. Keep expected CLI errors brief;
        # the Python API preserves the original exception and its cause.
        message = " ".join(str(exc).split())
        if len(message) > 1000:
            message = message[:997] + "..."
        print(f"{parser.prog}: error: {message}", file=sys.stderr)
        return 1
    return 0
