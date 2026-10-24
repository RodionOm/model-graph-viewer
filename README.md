# Model Graph Viewer

A Python library and command-line tool for inspecting neural network computational
graphs. The pipeline loads an OpenVINO IR model, builds a framework-independent
graph, applies ordered visualization passes, and exports an SVG.

Development follows the requirements in [assignment.pdf](assignment.pdf).
The [implementation plan](IMPLEMENTATION_PLAN.md) tracks each stage and its acceptance
checks.

## Current status

The package setup, environment check, framework-independent graph API, OpenVINO
IR loader, transformation pipeline, highlighting, and constant hiding are
implemented and tested, together with port-aware SVG rendering, a public Python
API, and the model-to-SVG CLI.
All ten implementation stages are complete, with the limitations documented below.
The current suite passes
254 tests, including real OpenVINO-to-SVG and API/CLI integration cases. Lint and formatting checks
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
optional for users exporting their own models with Optimum Intel; it is not needed
for the checked examples. TinyLlama is downloaded as a pinned, pre-exported IR
using the standard-library script below. No PyTorch/export stack is required.

Each completed implementation stage is tested, committed separately, and pushed
to this repository. Generated local models, virtual environments, and temporary
artifacts are excluded from Git.

The integration tests generate small XML/BIN models in pytest's temporary
directory. They require OpenVINO; SVG integration tests also require native
Graphviz on `PATH`. They do not download models or require a GPU or model
compilation. Run only the tests without native Graphviz or OpenVINO with
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

The script writes `models/linear.xml` and `models/linear.bin`, plus a branching
model in `models/branching.xml` and `.bin`, and a two-layer MLP in `models/mlp.xml`
and `.bin`. It can be rerun to
regenerate the example; these generated files are ignored by Git. Its weights are
`W = [[0.5], [1.0]]` and `b = [1.0]`. With input `[[2, 3]]`, the output is `[[5.0]]`,
which has also been checked with OpenVINO CPU inference.

## Command-line usage

After generating the models above, render the original or transformed graph.
Native Graphviz must be on `PATH`, as described in Setup.

Windows PowerShell:

```powershell
.\.venv\Scripts\python -m model_graph_viewer --model models/linear.xml --output artifacts/original.svg
.\.venv\Scripts\python -m model_graph_viewer `
    --model models/linear.xml --output artifacts/simplified.svg `
    --passes highlight,hide-constants --highlight-node matmul
```

Linux or macOS:

```sh
.venv/bin/python -m model_graph_viewer --model models/linear.xml --output artifacts/original.svg
.venv/bin/python -m model_graph_viewer \
    --model models/linear.xml --output artifacts/simplified.svg \
    --passes highlight,hide-constants --highlight-node matmul
```

With the virtual environment active, `model-graph-viewer` is equivalent to
`python -m model_graph_viewer`.

- `--model` and `--output` are both required for rendering. The output must end
  in `.svg`; parent directories are created. Quote paths containing spaces.
- Without `--passes`, no transformations run. Supply a comma-separated sequence
  of `highlight` and `hide-constants`. Order is preserved, including repeated
  passes. Repeated `--passes` arguments append to that sequence.
- `--highlight-node` is repeatable and matches exact, case-sensitive names,
  including all duplicate names. For example, `--highlight-node "layer, one"
  --highlight-node "layer two"` selects two complete names. Commas in names are
  never split; surrounding spaces are significant. Unknown names match nothing.
- The `highlight` pass requires at least one `--highlight-node`; supplying names
  without that pass is an argument error. Disable a pass by omitting it from
  `--passes`. For independent options and enabled flags on every step, use
  `--pass-config` as described in Extensibility below.
- `--weights path/to/weights.bin` selects weights separately from the XML.
- `--max-label-length 200` changes ordinary node field truncation (minimum 8).
  `--timeout 60` limits Graphviz layout to 60 seconds (positive, finite values).
- `--layout fast` bounds Graphviz optimization and uses straight edges for large
  graphs. The default uses Graphviz's normal optimization and curved routing.
- `--check-environment` runs the setup check separately and cannot be combined
  with rendering options. `--help`, `--version`, and a no-argument help invocation
  do not need the backend packages or native Graphviz.

Successful rendering prints the output path and exits with code 0. Invalid
arguments exit with code 2. Expected loading, transformation, dependency, and
rendering failures print a short message to stderr and exit with code 1.
Unexpected programming errors retain their traceback. A failed render preserves
an existing SVG, and no model or weight file is modified.

## Python API

The complete pipeline is available from the package root:

```python
from model_graph_viewer import visualize
from model_graph_viewer.passes import HideConstantsPass, HighlightPass

output = visualize(
    "models/linear.xml",
    "artifacts/api.svg",
    passes=[HighlightPass(["matmul"]), HideConstantsPass()],
)
print(output)  # pathlib.Path
```

`visualize(model_path, output_path, *, passes=(), weights_path=None, loader=None,
max_label_length=120, timeout=None, layout="default")` applies no transformations by default.
Pass objects, custom passes, `PassStep` objects, and generators are supported.
Steps execute in order, including repeats. Source model files are never changed.

Use separate operations to inspect or modify the intermediate graph:

```python
from model_graph_viewer import apply_passes, load_graph, render_graph
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, PassStep

graph = load_graph("models/linear.xml")
apply_passes(graph, [
    HighlightPass(["matmul"]),
    PassStep(HideConstantsPass(), enabled=False),
])
print(len(graph.nodes))  # 6: constant hiding was disabled
render_graph(graph, "artifacts/api_separate.svg", timeout=60)
```

`load_graph` accepts optional `weights_path` or a configured `loader` instance; `render_graph` accepts
`max_label_length`, `timeout`, and `layout`, as above. `apply_passes` changes the supplied
graph in place and returns it. It validates the graph even for an empty pass
sequence. `render_graph` returns the output `Path` and does not modify the graph.
Imports from `model_graph_viewer` do not import OpenVINO or Graphviz; each backend
is loaded only when needed.

Expected backend failures raise subclasses of `model_graph_viewer.errors.ViewerError`
with full context and exception causes. Invalid Python configuration types or
values raise `TypeError` or `ValueError`; unexpected programming errors propagate.
Completed pass changes are not rolled back on failure. The lower-level loader,
pipeline, and renderer interfaces below remain available for more control.

## Loader details

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
and dtype. Render this graph with the [Python API](#python-api).

`loader.read_model(path)` returns an OpenVINO model; `loader.parse(model)` converts
an existing model to our graph. `load(path)` combines these steps. Reading uses
[OpenVINO Core.read_model](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Core.html)
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
  the graph representation, transformation passes, or renderer.

## Apply visualization transformations

Passes operate on the independent graph. They modify it in place, without copying
the whole graph or touching the original OpenVINO model. Each pass implements
`apply(graph) -> None` or returns the same graph; `Pipeline.apply(graph)` returns
the same graph for convenience. Returning a different graph is rejected.

```python
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HighlightPass, PassStep, Pipeline

graph = OpenVINOLoader().load("models/linear.xml")
pipeline = Pipeline([
    HighlightPass(["matmul", "weights"]),
    PassStep(HighlightPass(["add"]), enabled=False),
])
pipeline.apply(graph)

for node in graph.nodes.values():
    if node.visualization.get("border_color") == "red":
        print(node.name)  # weights and matmul
```

`HighlightPass` matches complete, case-sensitive node names. It highlights every
matching node, including duplicate names. Spaces, commas, and Unicode characters
are literal; names are not split or treated as patterns. Unknown names and empty
selections leave the graph unchanged. Pass a collection of names such as
`["matmul"]`, not a bare string.

Highlighting sets `node.visualization["border_color"] = "red"`. It preserves all
other visualization metadata, operation attributes, ports, and connections.
Repeated highlighting is safe, and selecting more names later does not clear
previous highlights. The SVG renderer uses this metadata to draw red borders
while preserving the node fill color.

Passes run in the order supplied. Plain pass objects are enabled; wrap any pass in
`PassStep(..., enabled=False)` to skip it without changing its implementation.
`Pipeline()` applies no transformations. Repeated entries are executed repeatedly,
and the pipeline snapshots the supplied sequence so generators can also be used.

The pipeline validates the graph initially and after each enabled pass. An invalid
graph stops execution with `GraphValidationError`, including the failing pass's
position and class when applicable. Unexpected programming errors propagate.
Completed changes are not rolled back on failure; reload the model to start over.
Returning a replacement graph from a pass raises `TypeError` because passes use
the in-place contract; returning the supplied graph is accepted. Validation costs one graph traversal per enabled pass,
and highlighting scans the nodes with a set lookup for each name.

### Hide constant nodes

`HideConstantsPass` replaces outgoing Constant connections with records on their
consumers. Each record preserves the destination input port, Constant ID, source
output port, and tensor dtype and shape. The original operation attributes and
unrelated connections are preserved.

```python
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline

graph = OpenVINOLoader().load("models/linear.xml")
Pipeline([HighlightPass(["matmul"]), HideConstantsPass()]).apply(graph)
print(f"{len(graph.nodes)} nodes, {len(graph.edges)} edges")  # 4 nodes, 3 edges

for node in graph.nodes.values():
    for record in node.visualization.get("hidden_constants", {}).values():
        print(node.name, str(record))
# matmul [1]: fp32[2, 1]
# add [1]: fp32[1]
```

The records are immutable `ConstantInput` objects, available from
`model_graph_viewer`, under the reserved visualization key `hidden_constants`.
Its value is a dictionary keyed by destination input port, for example
`node.visualization["hidden_constants"][1]`. This is separate from
`node.attributes`, so an existing operation attribute with the same name or
port label is not overwritten. The renderer shows each record as a separate
line inside the consumer node.

If a Constant feeds multiple distinct consumer nodes, every record includes the
same marker, such as `[1]: fp32[2, 1] (shared: 4)`. The marker uses its unique ID,
not its friendly name. A single operation using the same Constant on two inputs
receives two records without the shared marker. Consumer counts are computed
before any edge is removed.

Only the hidden Constant edges are removed. A Constant node is removed only once
it has no remaining visible connections. A direct Constant-to-Result connection
leaves the Result node with its input record. Missing metadata, dynamic shapes,
and scalars retain the same representation as other tensors. Tensor values are
never read, copied, or displayed by the pass.

Repeated execution does not duplicate records. Existing records on other input
ports are preserved; incompatible `hidden_constants` data or an already-annotated
port with a new Constant edge raises `GraphValidationError` before this pass makes
any changes. Adjacency indexes and metadata snapshots keep work proportional to
nodes, Constant edges, and existing annotations, without copying the whole graph.

Highlighting composes with hiding in either order. Highlighted consumer nodes
keep their border metadata; a highlighted Constant is still hidden. Tests verify
that both serialized model structure and weight bytes remain unchanged.

### Add a transformation

Implement `apply` with the same contract. Inheritance and registration are not
required; the `TransformationPass` protocol describes this interface for type
checking. For example:

```python
from model_graph_viewer import Graph


class MarkOutputs:
    def apply(self, graph: Graph) -> None:
        for node in graph.nodes.values():
            if node.op_type == "Result":
                node.visualization["border_color"] = "blue"
```

Pass `MarkOutputs()` to `Pipeline` like a built-in pass. A following `HighlightPass`
can change the selected output borders to red. Adding this pass requires no changes
to the loader or graph representation. Graph and pass imports do not load OpenVINO
or Graphviz.

## Extensibility

Passes and model adapters can be added without editing API, CLI, graph, or renderer
code. There is no global plugin registry or automatic discovery. Python callers
pass objects; CLI callers explicitly select an importable `module:factory`.

For a new format, implement `load(model_path) -> Graph` and pass an instance:

```python
from model_graph_viewer import visualize
from examples.extensions import JsonGraphLoader

visualize("examples/custom_graph.json", "artifacts/custom.svg", loader=JsonGraphLoader())
```

The protocol does not require `weights_path`. This option is specific to the
default OpenVINO loader; when using a custom loader, configure its additional
options on the instance or in its zero-argument factory. An equivalent CLI call
from the repository root is:

```sh
python -m model_graph_viewer --model examples/custom_graph.json --output artifacts/custom.svg --loader examples.extensions:JsonGraphLoader
```

This small JSON adapter is an extension demonstration, not ONNX support. It reads
nodes/edges without tensor metadata. For ONNX, implement its parsing in a new
adapter returning the same `Graph` contract and select that adapter in the same way.

For per-step options, pass a JSON array using `--pass-config`:

```json
[
  {"name": "highlight", "options": {"node_names": ["matmul"]}},
  {"name": "hide-constants"},
  {"name": "highlight", "options": {"node_names": ["add"]}, "enabled": false},
  {"name": "examples.extensions:MarkOutputs", "options": {"color": "blue"}}
]
```

```sh
python -m model_graph_viewer --model models/linear.xml --output artifacts/configured.svg --pass-config examples/passes.json
```

Each entry has its own constructor options and an optional boolean `enabled`
(default `true`). Entries are applied in array order. A name is a built-in alias
or an importable factory returning a pass with `apply(graph)`. The example custom
pass is in [examples/extensions.py](examples/extensions.py). Its only dependency
is the graph API. Configured factories are instantiated and validated even when
the step is disabled; disabled steps are not applied.

`--pass-config` cannot be combined with `--passes` or `--highlight-node`; the
short CLI form retains its shared highlight-name list. Factories are ordinary
Python code: selecting one imports and executes it. Use code you intend to run,
and install the module or make it importable on `PYTHONPATH`.

## Render SVG

The renderer accepts the independent graph, including graphs created without
OpenVINO. With native Graphviz on `PATH`:

```python
from model_graph_viewer.loaders.openvino import OpenVINOLoader
from model_graph_viewer.passes import HideConstantsPass, HighlightPass, Pipeline
from model_graph_viewer.renderers import GraphvizRenderer

graph = OpenVINOLoader().load("models/linear.xml")
renderer = GraphvizRenderer()
renderer.render(graph, "artifacts/linear_before.svg")
Pipeline([HighlightPass(["matmul"]), HideConstantsPass()]).apply(graph)
output = renderer.render(graph, "artifacts/linear_after.svg")
print(output)
```

`render(graph, path)` creates parent directories and returns the output `Path`.
`render_svg(graph)` returns SVG bytes without writing a file. `to_dot(graph)`
returns a Python `graphviz.Digraph` for inspection without invoking native `dot`.
None of these methods changes the graph or imports OpenVINO.

Nodes show their operation type, name, ID, operation attributes, input/output
ports, and hidden-constant records. An unconnected backend output on `Result` is
omitted from its label, while remaining available in `Graph`; a connected output
is always drawn. Edges connect explicit ports and show both
port numbers and tensor metadata. Layout runs from top to bottom. Unknown
operation types use a neutral fill; a pass can set `visualization["fillcolor"]`,
`["border_color"]`, or `["note"]`. An empty graph produces an empty SVG canvas;
an isolated node renders normally.

Ordinary node fields are shortened to 120 characters by default; set
`GraphvizRenderer(max_label_length=200)` to change this. Original graph data and
complete port/tensor records are retained. Containers are formatted with bounded
depth and length, opaque attribute objects use type placeholders, and Constant
nodes show only dtype/shape metadata. Tensor values are not displayed. Label text
is escaped, including Graphviz backslash substitutions; XML-invalid control
characters are replaced with the Unicode replacement character.

Missing `dot`, a failed layout, invalid SVG output, and write failures raise
`model_graph_viewer.errors.RenderError`. An optional `timeout=60` limits the
native layout process to 60 seconds; the default has no time limit. A file is
replaced atomically only after valid SVG has been produced, so a failed render
preserves the existing output. Graphviz warnings are sent to Python logging.
Exact coordinates and font metrics can vary by Graphviz version and system fonts.

`GraphvizRenderer(layout="fast")` retains every node, port, connection and label,
while setting `nslimit=0.5`, `nslimit1=0.5`, `mclimit=0.25`, and `splines=line`.
These limit optimization effort and avoid costly spline routing; they are not a
wall-clock guarantee. Straight edges can cross nodes and labels. Use normal
layout for small graphs and the fast mode when a full large graph is needed.
See Graphviz's [iteration limits](https://graphviz.org/docs/attrs/nslimit/),
[rank optimization](https://graphviz.org/docs/attrs/nslimit1/),
[crossing minimization](https://graphviz.org/docs/attrs/mclimit/), and
[edge routing](https://graphviz.org/docs/attrs/splines/) documentation.

### View and reproduce examples

Eight checked SVGs are committed in [examples/svg](examples/svg):

| Model | Before | After highlighting and constant hiding |
| --- | --- | --- |
| Linear: 6 → 4 nodes, 5 → 3 edges | [Before](examples/svg/linear_before.svg) | [After](examples/svg/linear_after.svg) |
| Branching: 9 → 7 nodes, 10 → 7 edges | [Before](examples/svg/branching_before.svg) | [After](examples/svg/branching_after.svg) |
| MLP: 11 → 7 nodes, 10 → 6 edges | [Before](examples/svg/mlp_before.svg) | [After](examples/svg/mlp_after.svg) |
| TinyLlama: 3216 → 1957 nodes, 3789 → 2477 edges | [Before](examples/svg/tinyllama_before.svg) | [After](examples/svg/tinyllama_after.svg) |

The branching example has two Split outputs, parallel connections into an Add,
a Constant shared between two consumers, and two model outputs. Both examples
are deterministic, untrained OpenVINO models. The MLP is a fixed-weight 4 → 3 → 2
network with ReLU. TinyLlama is a real pretrained 1.1B-parameter language model.

Generate the three small models and their six SVGs (with the virtual environment active and
`dot` on `PATH`; these commands work in PowerShell and POSIX shells):

```sh
python scripts/create_example_models.py --output-dir models
python scripts/render_examples.py --model-dir models --output-dir examples/svg
```

Open SVG files in Edge, Chrome, or Firefox. If the editor displays XML text, drag
the file into a browser window. In Windows PowerShell, you can also run:

```powershell
Start-Process msedge (Resolve-Path examples/svg/linear_after.svg).Path
```

### TinyLlama reproduction

The large example uses [OpenVINO's fp16 TinyLlama IR](https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov),
converted from `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, at revision
`88a9e6bbe7f9f8cf4e61ac96eed57ec164ad46f5`. The assignment allows downloading
already exported models. The script downloads 2.20 GB of weights plus XML/config
and the source model card; it verifies sizes and SHA256 against
[tinyllama-source.json](examples/tinyllama-source.json). Already verified files
are reused. Model files remain in the ignored `models/` directory.

With the virtual environment active and Graphviz on `PATH`, these single-line
commands work in PowerShell and POSIX shells:

```sh
python scripts/download_tinyllama.py
python -m model_graph_viewer --model models/tinyllama/openvino_model.xml --output examples/svg/tinyllama_before.svg --layout fast --timeout 180
python -m model_graph_viewer --model models/tinyllama/openvino_model.xml --output examples/svg/tinyllama_after.svg --layout fast --timeout 180 --passes highlight,hide-constants --highlight-node "__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul"
```

The model's IR metadata records OpenVINO `2026.0.0-20961-f08e8b36350`,
Optimum Intel `1.27.0.dev0+2c48d64`, Optimum `2.1.0.dev0`, PyTorch `2.10.0+cpu`,
and Transformers `4.57.6`. These describe the upstream export, not dependencies
needed to visualize it. Our checked runtime is OpenVINO 2026.4.1. We reproduce
the downloaded IR, not the upstream export environment.

The graph contains 1259 Constants, 433 Converts, 44 ReadValues, 44 Assigns and
22 ScaledDotProductAttention nodes, among other operations. After hiding
Constants, Convert nodes remain. Both SVGs have been parsed to verify every
expected node/edge and reviewed in Edge at a selected projection. The complete
SVGs are very tall; zoom is essential and the fast layout has visible crossings.

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
Names must be strings (empty names are allowed); operation types must be nonempty
strings. Nodes have value equality and are explicitly unhashable because their
metadata is mutable. Use `node_id` as a dictionary/set key.

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

## Architecture

```text
OpenVINO XML + BIN -> Loader -> Graph -> Transformation pipeline -> Renderer -> SVG
                                             ^
                                             |
                                   Highlight / Hide constants
```

The graph owns ordinary Python data, with unique node IDs independent of
names and edges identifying both source and destination ports. The OpenVINO
loader is isolated from this representation. Passes modify the
visualization graph without modifying the original model. The CLI calls the
same API available to Python users.

## Verification

On 2026-10-24 the final code was built as a wheel and installed with the pinned
dependencies in a separate virtual environment, without an editable installation.
Imports were verified to come from `site-packages`. All 254 tests passed with no
skips; Ruff lint and formatting passed. The environment check, nine standalone
Python examples from this README, installed CLI, custom loader/pass examples,
model generators, downloader verification and benchmark scripts were exercised.

All eight SVGs were regenerated using that installation and matched the committed
working-tree examples byte for byte on this Windows/Graphviz/font environment.
The final wheel's 18 Python source files were compared with the repository.
[verification.json](benchmarks/verification.json) records checks and SVG SHA256.
Model weights, local environments, diagnostic previews and temporary artifacts
are excluded from Git. Linux/macOS commands are provided as equivalents, but
those platforms were not tested; SVG byte identity is not promised across fonts
or Graphviz versions.

## Performance

Measured on 2026-10-24: Intel Core i5-1240P (16 logical CPUs), OS-reported
7.62 GiB RAM, Windows 11 build 26200, Python 3.12.10, OpenVINO 2026.4.1,
Python graphviz 0.21, native Graphviz 16.1.0. No model compilation or inference
was performed. The environment and full measurements are saved in
[tinyllama.json](benchmarks/tinyllama.json),
[small-and-scale.json](benchmarks/small-and-scale.json), and
[scaling.json](benchmarks/scaling.json).

Times below are seconds. Load includes reading IR and constructing the independent
graph; the JSON also separates these. Pass time includes validation. Render
includes DOT generation, native layout, SVG validation and file writing.

| Model / variant | Nodes | Edges | Load | Passes | Render |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP / original | 11 | 10 | 0.0094 | — | 0.1792 |
| MLP / transformed | 7 | 6 | 0.0036 | 0.0002 | 0.4800 |
| Branching / original | 9 | 10 | 0.0037 | — | 0.4701 |
| Branching / transformed | 7 | 7 | 0.0038 | 0.0001 | 0.5825 |
| TinyLlama / original, fast layout | 3216 | 3789 | 0.3130 | — | 25.4523 |
| TinyLlama / transformed, fast layout | 1957 | 2477 | 0.3254 | 0.0252 | 13.3849 |
| Synthetic ReLU chain / original | 5002 | 5001 | 0.3352 | — | 2.0729 |
| Synthetic ReLU chain / transformed | 5002 | 5001 | 0.2936 | 0.0411 | 1.9440 |

These are one measured run per variant, with the original first, not averages.
Files were hashed before timing, so OS file caching was warm. Download/export and
hash verification, process startup and imports are excluded. The synthetic chain contains no Constants;
its transformed case exercises validation/highlighting without reducing the graph.
It is a scale fixture, not a substitute for the real TinyLlama run. The fast
layout changes geometry/routing only. The small/chain cases use default layout.

The main bottleneck on TinyLlama is Graphviz. Default layout did **not finish
within 600 seconds**; a bounded polyline-routing trial also timed out at 120
seconds. These are unsuccessful trials, recorded in
[layout-diagnostics.json](benchmarks/layout-diagnostics.json), not successful
render times. Verbose Graphviz output showed 422,151 internal positioning nodes
and 631,894 internal edges created around long-range connections. Limiting
optimization alone was insufficient; straight routing made the complete SVG
practical, at the cost of crossings through nodes/labels. Some diagnostic trials
overlapped other checks; they should not be treated as controlled speed ratios.

Separate processing-only measurements use an Add chain sharing one Constant
(three runs per size, medians; no rendering):

| Nodes before | Edges before | Construct graph | Highlight + hide |
| ---: | ---: | ---: | ---: |
| 1002 | 2000 | 0.0086 | 0.0072 |
| 5002 | 10000 | 0.0411 | 0.0549 |
| 10002 | 20000 | 0.1187 | 0.1150 |

This supports approximately linear graph/pass processing over these sizes.
Adjacency indexes avoid full-graph searches per edge; passes operate in place
and carry shape/type records, not tensor values. The viewer does not copy the
model weights. Native OpenVINO/Graphviz storage is separate: a process snapshot
during the failed default layout showed roughly 0.59 GB of `dot` working set.
This is **not a peak-memory measurement** or total application memory. Python-only
memory counters would miss native allocations and mapped weight storage.

Reproduce measurements explicitly (large jobs are never part of ordinary pytest):

```sh
python scripts/create_example_models.py
python scripts/download_tinyllama.py
python scripts/benchmark_scaling.py
python scripts/benchmark.py --model models/mlp.xml --model models/branching.xml --model models/synthetic_5002.xml --output-dir artifacts/benchmark-svg --report benchmarks/small-and-scale.json
python scripts/benchmark.py --model models/tinyllama/openvino_model.xml --layout fast --timeout 180 --report benchmarks/tinyllama.json
```

The benchmark saves a failed status and error, without a claimed SVG, if rendering
fails; it returns exit code 1. Successful timings vary with machine load, caches,
fonts and Graphviz versions. Future improvements could expose focused subgraphs
or collapse carefully defined patterns to reduce long-range edges; these are not
implemented, and would need separate correctness tests.

## Known limitations

- Only OpenVINO IR is built in. Other formats require an adapter; no model
  compilation or inference is performed by the viewer.
- `If`, `Loop`, and `TensorIterator` bodies are not expanded. Only outer data
  connections are represented; state-variable relationships and execution/control
  dependencies are not extra data edges. Symbolic equality of dimensions is not retained.
- Constant hiding stops at the immediate consumer. In a fp16
  `Constant -> Convert -> MatMul` chain, the weight record appears on `Convert`;
  the conversion node remains visible. This pass does not collapse such patterns.
- A highlighted Constant disappears when hidden. "Shared" counts distinct
  consumer operations, so one operation using a Constant twice has two records
  without a shared marker. Both behaviors are deliberate and tested.
- Ordinary node fields are shortened; tensor/port records remain complete.
  Backend operation attributes may have different semantics from edge metadata;
  declared variable shapes are not replaced by inferred output shapes.
- Large SVGs need browser zoom and can be difficult to navigate. Graphviz layout
  cost depends strongly on topology, not just node count. A fast chain benchmark
  does not predict performance on LLMs. Exact layout varies with fonts/versions.
- Passes modify the visualization graph in place without rollback. They may
  return `None` or the same graph, but cannot replace it with a new graph.
