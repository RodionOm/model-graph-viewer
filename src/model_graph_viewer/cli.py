"""Command-line entry point; model visualization is added in later stages."""

import argparse
import sys
from collections.abc import Sequence

from model_graph_viewer import __version__
from model_graph_viewer.environment import check_environment
from model_graph_viewer.errors import ViewerError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-graph-viewer",
        description="Inspect setup for the neural network graph viewer.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="check OpenVINO and Graphviz, including a small SVG render",
    )
    args = parser.parse_args(argv)
    if args.check_environment:
        try:
            for name, value in check_environment().items():
                print(f"{name}: {value}")
        except ViewerError as exc:
            print(f"{parser.prog}: error: {exc}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
    return 0
