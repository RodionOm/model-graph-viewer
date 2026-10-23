# Model Graph Viewer

A Python library and command-line tool for inspecting neural network computational
graphs. The planned pipeline loads an OpenVINO IR model, builds a framework-independent
graph, applies ordered visualization passes, and exports an SVG.

Development follows the requirements in [assignment.pdf](assignment.pdf).
The [implementation plan](IMPLEMENTATION_PLAN.md) tracks each stage and its acceptance
checks.

## Current status

The package setup, environment check, framework-independent graph API, and
OpenVINO IR loader are implemented and tested. Transformation passes and the
model-to-SVG command are still in development.
Stages 1–3 of the implementation plan are complete. The current suite passes
88 tests, including 35 OpenVINO integration cases. Lint and formatting checks
also pass. Earlier setup checks verified the Graphviz SVG backend and installation
of a built wheel independently of the editable source checkout.
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
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

The `dev` extra contains test and lint tools. The separate `export` extra is
reserved for exporting large models with Optimum Intel; it is not required for
unit tests or graph inspection. The TinyLlama export procedure will be documented
and checked when the model examples are implemented.

Each completed implementation stage is tested, committed separately, and pushed
to this repository. Generated local models, virtual environments, and temporary
artifacts are excluded from Git.

The integration tests generate small XML/BIN models in pytest's temporary
directory. They require OpenVINO but do not download models or require Graphviz,
a GPU, or model compilation. Run only the framework-independent tests with
`python -m pytest tests/unit`, or the loader tests with
`python -m pytest tests/integration/test_openvino_loader.py`.

## Load a real OpenVINO model

Generate a small model with fixed fp32 weights. This is a real, executable
OpenVINO model of `y = x @ W + b`, created for testing rather than trained on data.

Windows PowerShell:

```powershell
.\.venv\Scripts\python scripts/create_example_models.py --output-dir models
```

Linux or macOS:

```sh
.venv/bin/python scripts/create_example_models.py --output-dir models
```

The script writes `models/linear.xml` and `models/linear.bin`. It can be rerun to
regenerate the example; these generated files are ignored by Git. Its weights are
`W = [[0.5], [1.0]]` and `b = [1.0]`. With input `[[2, 3]]`, the output is `[[5.0]]`,
which has also been checked with OpenVINO CPU inference.

Use the loader from Python:

```python
from model_graph_viewer.loaders.openvino import OpenVINOLoader

loader = OpenVINOLoader()
graph = loader.load("models/linear.xml")
print(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")  # 6 nodes, 5 edges

for node in graph.nodes.values():
    print(node.node_id, node.name, node.op_type)
```

The operations are `Parameter`, two `Constant` nodes, `MatMul`, `Add`, and `Result`.
Weights remain in the model: the intermediate graph contains only their shape
and dtype. SVG export of this graph will be added in the rendering stage.

`loader.read_model(path)` returns an OpenVINO model; `loader.parse(model)` converts
an existing model to our graph. `load(path)` combines these steps. Reading uses
[OpenVINO Core.read_model](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.Core.html)
without compiling, reshaping, or optimizing the model.

By default, the loader looks for a `.bin` next to the `.xml` with the same stem.
A model without weights can omit the `.bin`. To select a different weights file,
use `loader.load("model.xml", weights_path="weights.bin")`. Unsupported extensions,
missing files, invalid XML, and invalid weight files raise
`model_graph_viewer.errors.ModelLoadError` with the affected path and a reason.

### Loader guarantees and limits

- IDs are assigned from OpenVINO's ordered operations, independently of friendly
  names. Re-reading the same IR with the same runtime is tested to preserve IDs;
  IDs are not promised to survive model edits or OpenVINO version changes.
- All visible input/output port indices, fan-out, parallel connections, and model
  inputs and outputs are retained. Declared ports survive even without connections.
- Scalar shapes, dynamic rank, dynamic dimensions, and dimension intervals are
  represented separately. Unresolved element types become `None`; f16, f32, and
  f64 are displayed as fp16, fp32, and fp64. Other type names are preserved.
- Operation attributes are detached as ordinary Python values. Unknown non-scalar
  attribute objects receive a descriptive placeholder instead of retaining a
  framework object or reading a tensor. Constant values are never requested.
- Modifying the resulting graph does not modify the input model. Tests compare
  both serialized model structure and weight bytes before and after graph edits.
- `If`, `Loop`, and `TensorIterator` are represented by their outer nodes and ports,
  with a note in visualization metadata. Nested bodies are not expanded. Dimension
  bounds are retained, but symbolic equality relationships between dimensions are
  not represented.
- Only OpenVINO IR is accepted. A new format can implement
  `model_graph_viewer.loaders.ModelLoader.load(path) -> Graph` without changing
  the graph representation or future passes and renderers.

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
names and edges identifying both source and destination ports. The OpenVINO
loader is isolated from this representation. Passes will modify the
visualization graph without modifying the original model. The CLI will call the
same API available to Python users.

## Remaining deliverables

- Highlighting, constant hiding, and the configurable transformation pipeline.
- Port-aware SVG rendering and configurable Python API and CLI.
- Unit and integration tests, including a complete model-to-SVG test.
- Three reproducible model examples, including TinyLlama-1.1B-Chat-v1.0.
- Before-and-after SVGs and measured performance, including graphs of 5000+ nodes.
- Final API documentation, extension examples, and known limitations.

Benchmark results and completed model visualizations will be added after they
have actually been produced and verified.
