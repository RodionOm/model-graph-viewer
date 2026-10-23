# Model Graph Viewer

A Python library and command-line tool for inspecting neural network computational
graphs. The planned pipeline loads an OpenVINO IR model, builds a framework-independent
graph, applies ordered visualization passes, and exports an SVG.

Development follows the requirements in [assignment.pdf](assignment.pdf).
The [implementation plan](IMPLEMENTATION_PLAN.md) tracks each stage and its acceptance
checks.

## Current status

The package setup, environment check, and framework-independent graph API are
implemented and tested. Model loading, transformation passes, and the model-to-SVG
command are still in development.
The pre-loader review on 23 October 2026 passed 53 tests, lint and formatting
checks, dependency checks, the README graph example, and the Graphviz SVG setup
check. A built wheel was also installed in a separate environment to verify the
graph API and CLI entry points independently of the editable source checkout.
The original [starter_code.py](starter_code.py) is preserved as a standalone
Graphviz example.

## Setup

Use Python 3.11 or newer. Development is being checked on Windows with Python 3.12.
Create a dedicated virtual environment in the repository.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m model_graph_viewer --help
```

Linux or macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m model_graph_viewer --help
```

Install the native Graphviz tools separately using the
[official installation instructions](https://graphviz.org/download/).
The Python `graphviz` package does not include the `dot` executable.
On Windows, either use the installer or extract the official ZIP and add the
directory containing `dot.exe` to the current shell's `PATH`:

```powershell
$env:PATH = "C:\path\to\Graphviz\bin;$env:PATH"
dot -V
.\.venv\Scripts\python -m model_graph_viewer --check-environment
```

On Linux or macOS, once `dot` is on `PATH`:

```sh
dot -V
.venv/bin/python -m model_graph_viewer --check-environment
```

The check imports OpenVINO and Python graphviz, reports their versions, and asks
Graphviz to generate a small SVG in memory. It reports missing dependencies with
a nonzero exit code.

After activating the virtual environment, the equivalent installed command is
`model-graph-viewer --check-environment`.

### Verified environment

The setup check and initial tests pass on Windows x64 with Python 3.12.10,
OpenVINO 2026.4.1, Python graphviz 0.21, Graphviz 16.1.0, and pytest 9.1.1.
To reproduce these Python dependency versions on Python 3.12, install with:

```sh
python -m pip install -c constraints-dev.txt -e ".[dev]"
```

Graphviz 16.1.0 was downloaded as the official Windows ZIP, checked against the
published SHA256, and extracted locally. The binaries are not committed.

## Development

Run the following with the virtual environment's Python:

```sh
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

The `dev` extra contains test and lint tools. The separate `export` extra is
reserved for exporting large models with Optimum Intel; it is not required for
unit tests or graph inspection. The TinyLlama export procedure will be documented
and checked when the model examples are implemented.

Each completed implementation stage is tested, committed separately, and pushed
to this repository. Generated local models, virtual environments, and temporary
artifacts are excluded from Git.

## Graph API

Construct and inspect a graph directly, without loading a model:

```python
from model_graph_viewer import Edge, Graph, Node, TensorMetadata, TensorShape

tensor = TensorMetadata(shape=TensorShape((None, 768)), dtype="fp32")
graph = Graph(
    nodes=[
        Node("0", "input", "Parameter", output_ports=(0,)),
        Node("1", "add", "Add", input_ports=(0, 1), output_ports=(0,)),
    ],
    edges=[
        Edge("0", 0, "1", 0, tensor),
        Edge("0", 0, "1", 1, tensor),
    ],
)
graph.validate()

for edge in graph.incoming("1"):
    print(f"out {edge.from_port} -> in {edge.to_port}: {edge.tensor}")
```

This example preserves both connections from the same source to separate input
ports. Each input port has at most one source. An output can have many consumers.
Port numbers are explicit and need not be contiguous.

`graph.nodes` is a read-only mapping keyed by unique string IDs. Names may repeat.
`graph.edges` is a read-only mapping keyed by `(destination_node_id, input_port)`.
Use `add_node`, `add_edge`, `remove_edge(destination_id, input_port)`, and
`remove_node` to change structure. Removing a node also removes its incident
edges; removing an edge preserves the declared ports.
Port arguments must be nonnegative integers; booleans and floating-point values
are rejected on both edge creation and removal without changing the graph.

`incoming` and `outgoing` return snapshots of local connections. Adding or removing
an edge uses indexed lookups; removing a node visits its incident edges.
`validate()` checks endpoints, port indices, and index consistency in time
proportional to nodes, edges, and declared ports.

Node identity, ports, edges, and tensor metadata are immutable. Each node has
separate mutable `attributes` and `visualization` dictionaries. Dictionary
containers are copied on construction; nested attribute values are not deep-copied.
This allows future passes to change display metadata without changing graph
identity or overwriting operation attributes.

### Tensor metadata

| Python value for shape | Meaning | Display with dtype `fp32` |
| --- | --- | --- |
| `None` | Shape metadata unavailable | `fp32[unknown]` |
| `TensorShape(None)` | Dynamic rank | `fp32[...]` |
| `TensorShape(())` | Scalar | `fp32[]` |
| `TensorShape((None, 768))` | Known rank, unknown first dimension | `fp32[?, 768]` |
| `TensorShape((Dimension(1, 8), 768))` | Inclusive dimension interval | `fp32[1..8, 768]` |

Import `Dimension` from `model_graph_viewer` when using intervals. Its default
upper bound is unbounded, so `Dimension(4)` displays as `4..?`. Unknown dtype is
represented by `None` and displayed as `?`. Zero-size dimensions are preserved.

The graph API uses only the Python standard library. Importing it does not import
OpenVINO or Graphviz. Structural errors raise
`model_graph_viewer.errors.GraphValidationError` before an invalid insertion
modifies the graph.

## Planned architecture

```text
OpenVINO XML + BIN -> Loader -> Graph -> Transformation pipeline -> Renderer -> SVG
                                             ^
                                             |
                                   Highlight / Hide constants
```

The graph owns ordinary Python data, with unique node IDs independent of
names and edges identifying both source and destination ports. Parsers will be
the only components aware of model frameworks. Passes will modify the
visualization graph without modifying the original model. The CLI will call the
same API available to Python users.

## Remaining deliverables

- OpenVINO loading, highlighting and constant hiding.
- Port-aware SVG rendering and configurable Python API and CLI.
- Unit and integration tests, including a complete model-to-SVG test.
- Three reproducible model examples, including TinyLlama-1.1B-Chat-v1.0.
- Before-and-after SVGs and measured performance, including graphs of 5000+ nodes.
- Final API documentation, extension examples, and known limitations.

Benchmark results and completed model visualizations will be added after they
have actually been produced and verified.
