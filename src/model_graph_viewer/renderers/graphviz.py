"""Hierarchical SVG rendering with explicit ports and atomic file output."""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from html import escape
from itertools import islice
from pathlib import Path
from xml.etree import ElementTree

import graphviz

from model_graph_viewer.errors import RenderError
from model_graph_viewer.graph import ConstantInput, Graph, Node

_FILLCOLORS = {
    "Parameter": "#e2e8f0",
    "Input": "#e2e8f0",
    "Constant": "#fef3c7",
    "MatMul": "#dbeafe",
    "Add": "#ede9fe",
    "Result": "#dcfce7",
    "Output": "#dcfce7",
}


def _html(text: str) -> str:
    # Graphviz substitutes backslash escapes even inside HTML-like labels.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = "".join(
        char
        if char == "\n"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
        else "\ufffd"
        for char in text
    )
    return escape(text).replace("\\", "&#92;&#92;").replace("\n", '<BR ALIGN="LEFT"/>')


class GraphvizRenderer:
    """Render a Graph without importing a model framework or changing its data.

    max_label_length limits ordinary node fields; port and tensor metadata remain
    complete. timeout limits the external dot process, or None allows long layouts.
    """

    def __init__(
        self, *, max_label_length: int = 120, timeout: float | None = None, layout: str = "default"
    ) -> None:
        if type(max_label_length) is not int or max_label_length < 8:
            raise ValueError("max_label_length must be an integer of at least 8.")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number or None.")
        self.max_label_length = max_label_length
        self.timeout = timeout
        if layout not in {"default", "fast"}:
            raise ValueError("layout must be 'default' or 'fast'.")
        self.layout = layout

    def _short(self, text: str) -> str:
        limit = self.max_label_length
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _value(self, value: object, depth: int = 0) -> str:
        """Bound container formatting and never stringify opaque tensor objects."""
        if value is None or isinstance(value, (str, bool, int, float)):
            return self._short(str(value))
        if depth >= 3:
            return "..."
        if isinstance(value, (list, tuple)):
            items = [self._value(item, depth + 1) for item in islice(value, 12)]
            if len(value) > 12:
                items.append("...")
            return self._short("[" + ", ".join(items) + "]")
        if isinstance(value, dict):
            items = [
                self._value(k, depth + 1) + ": " + self._value(v, depth + 1)
                for k, v in islice(value.items(), 12)
            ]
            if len(value) > 12:
                items.append("...")
            return self._short("{" + ", ".join(items) + "}")
        return f"<{type(value).__name__}>"

    @staticmethod
    def _ports(ports: tuple[int, ...], direction: str) -> str:
        if not ports:
            return ""
        cells = "".join(
            f'<TD PORT="{direction}{port}"><FONT POINT-SIZE="9">{direction} {port}</FONT></TD>'
            for port in ports
        )
        # Nested tables avoid a large least-common-multiple column count when
        # a node has different numbers of inputs and outputs.
        return (
            '<TR><TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f"<TR>{cells}</TR></TABLE></TD></TR>"
        )

    def _label(self, node: Node, *, show_outputs: bool = True) -> str:
        fill = node.visualization.get("fillcolor", _FILLCOLORS.get(node.op_type, "#f8fafc"))
        border = node.visualization.get("border_color", "#64748b")
        if not isinstance(fill, str) or not isinstance(border, str):
            raise RenderError(f"Node {node.node_id!r} colors must be strings.")
        width = "3" if "border_color" in node.visualization else "1"
        rows = [self._ports(node.input_ports, "in")]
        rows.append(
            f"<TR><TD><B>{_html(self._short(node.op_type))}</B><BR/>"
            f"{_html(self._short(node.name))}<BR/>"
            f'<FONT POINT-SIZE="9">ID: {_html(self._short(node.node_id))}</FONT></TD></TR>'
        )
        for key, value in node.attributes.items():
            # Synthetic graphs may contain tensor values even though our loader
            # excludes them. Constant rendering only allows descriptive metadata.
            if node.op_type == "Constant" and key not in {"dtype", "shape", "element_type"}:
                continue
            text = self._short(f"{key}: {self._value(value)}")
            rows.append(f'<TR><TD ALIGN="LEFT">{_html(text)}</TD></TR>')
        hidden = node.visualization.get("hidden_constants", {})
        if not isinstance(hidden, dict):
            raise RenderError(f"Node {node.node_id!r} has invalid hidden_constants metadata.")
        ports = set(node.input_ports)
        for port, record in hidden.items():
            if (
                type(port) is not int
                or port not in ports
                or not isinstance(record, ConstantInput)
                or record.input_port != port
            ):
                raise RenderError(f"Node {node.node_id!r} has an invalid hidden constant port.")
        for port in sorted(hidden):
            rows.append(f'<TR><TD ALIGN="LEFT">{_html(str(hidden[port]))}</TD></TR>')
        if "note" in node.visualization:
            rows.append(
                '<TR><TD ALIGN="LEFT"><I>'
                + _html(self._value(node.visualization["note"]))
                + "</I></TD></TR>"
            )
        if show_outputs:
            rows.append(self._ports(node.output_ports, "out"))
        return (
            f'<<TABLE BORDER="{width}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6" '
            f'COLOR="{escape(border)}" BGCOLOR="{escape(fill)}">' + "".join(rows) + "</TABLE>>"
        )

    def to_dot(self, graph: Graph) -> graphviz.Digraph:
        """Build DOT without running Graphviz; input data stays untouched."""
        graph.validate()
        dot = graphviz.Digraph(
            "model",
            engine="dot",
            format="svg",
            strict=False,
            graph_attr={
                "rankdir": "TB",
                "bgcolor": "white",
                "pad": "0.25",
                "nodesep": "0.35",
                "ranksep": "0.4",
                "charset": "UTF-8",
            },
            node_attr={"shape": "plain", "fontname": "Arial", "fontsize": "11"},
            edge_attr={
                "fontname": "Arial",
                "fontsize": "9",
                "color": "#64748b",
                "arrowsize": "0.65",
            },
        )
        if self.layout == "fast":
            # Bound dot's search on long-range LLM connections. All graph data and
            # ports are retained; only position optimization and edge routes change.
            dot.graph_attr.update(nslimit="0.5", nslimit1="0.5", mclimit="0.25", splines="line")
        names = {node_id: f"n{index}" for index, node_id in enumerate(graph.nodes)}
        for node in graph.nodes.values():
            name = names[node.node_id]
            # Result's unused backend output is not another computation step.
            # Retain the actual port in Graph, and show it if it has connections.
            show_outputs = node.op_type != "Result" or bool(graph.outgoing(node.node_id))
            dot.node(name, label=self._label(node, show_outputs=show_outputs), id=f"node_{name}")
        for index, edge in enumerate(graph.edges.values()):
            label = f"out {edge.from_port} → in {edge.to_port}\n{edge.tensor}"
            dot.edge(
                f"{names[edge.from_node]}:out{edge.from_port}:s",
                f"{names[edge.to_node]}:in{edge.to_port}:n",
                label=f"<{_html(label)}>",
                id=f"edge_{index}",
            )
        return dot

    def render_svg(self, graph: Graph) -> bytes:
        """Run dot and return valid SVG bytes; write no files."""
        dot = self.to_dot(graph)
        executable = shutil.which("dot")
        if executable is None:
            raise RenderError(
                "Graphviz executable 'dot' was not found. Install Graphviz and "
                "add its bin directory to PATH; pip's graphviz package is not enough."
            )
        try:
            result = subprocess.run(
                [executable, "-Tsvg"],
                input=dot.source.encode("utf-8"),
                capture_output=True,
                check=True,
                timeout=self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError(f"Graphviz exceeded the {self.timeout}-second timeout.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", errors="replace")[-2000:]
            raise RenderError(f"Graphviz failed (exit {exc.returncode}): {detail}") from exc
        except OSError as exc:
            raise RenderError(f"Cannot execute Graphviz: {exc}") from exc
        if result.stderr:
            logging.getLogger(__name__).warning(
                "Graphviz: %s", result.stderr.decode("utf-8", errors="replace").strip()
            )
        try:
            if ElementTree.fromstring(result.stdout).tag != "{http://www.w3.org/2000/svg}svg":
                raise RenderError("Graphviz output is not an SVG document.")
        except ElementTree.ParseError as exc:
            raise RenderError("Graphviz returned invalid XML instead of SVG.") from exc
        return result.stdout

    def render(self, graph: Graph, output_path: str | Path) -> Path:
        """Write SVG atomically, preserving an existing output on failure."""
        path = Path(output_path)
        if path.suffix.lower() != ".svg":
            raise RenderError(f"Output path must end with .svg: {path}")
        svg = self.render_svg(graph)
        temporary: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(svg)
            os.replace(temporary, path)
        except OSError as exc:
            raise RenderError(f"Cannot write SVG to {path}: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logging.getLogger(__name__).warning(
                        "Cannot remove temporary SVG %s", temporary, exc_info=True
                    )
        return path
