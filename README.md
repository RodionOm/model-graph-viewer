# Model Graph Viewer

A Python library and command-line tool for inspecting neural network computational
graphs. The planned pipeline loads an OpenVINO IR model, builds a framework-independent
graph, applies ordered visualization passes, and exports an SVG.

Development follows the requirements in [assignment.pdf](assignment.pdf).
The [implementation plan](IMPLEMENTATION_PLAN.md) tracks each stage and its acceptance
checks.

## Current status

The package skeleton and environment check are implemented. Model loading,
transformation passes, and the model-to-SVG command are still in development.
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

## Planned architecture

```text
OpenVINO XML + BIN -> Loader -> Graph -> Transformation pipeline -> Renderer -> SVG
                                             ^
                                             |
                                   Highlight / Hide constants
```

The graph will own ordinary Python data, with unique node IDs independent of
names and edges identifying both source and destination ports. Parsers will be
the only components aware of model frameworks. Passes will modify the
visualization graph without modifying the original model. The CLI will call the
same API available to Python users.

## Remaining deliverables

- OpenVINO loading, graph representation, highlighting and constant hiding.
- Port-aware SVG rendering and configurable Python API and CLI.
- Unit and integration tests, including a complete model-to-SVG test.
- Three reproducible model examples, including TinyLlama-1.1B-Chat-v1.0.
- Before-and-after SVGs and measured performance, including graphs of 5000+ nodes.
- Final API documentation, extension examples, and known limitations.

Benchmark results and completed model visualizations will be added after they
have actually been produced and verified.
