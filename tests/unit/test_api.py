import subprocess
import sys

import pytest

from model_graph_viewer import Graph, Node, api, apply_passes, visualize
from model_graph_viewer.passes import PassStep


def test_public_pass_api_preserves_order_disabled_steps_and_repetitions():
    graph = Graph([Node("0", "input", "Parameter", attributes={"value": 1})])

    class Increment:
        def apply(self, graph):
            graph.nodes["0"].attributes["value"] += 1

    class Double:
        def apply(self, graph):
            graph.nodes["0"].attributes["value"] *= 2

    steps = [Increment(), Double(), PassStep(Double(), enabled=False), Increment()]
    assert apply_passes(graph, (step for step in steps)) is graph
    assert graph.nodes["0"].attributes["value"] == 5
    assert apply_passes(graph) is graph
    assert graph.nodes["0"].attributes["value"] == 5


def test_visualize_stops_before_rendering_on_pass_failure(monkeypatch):
    graph = Graph()
    monkeypatch.setattr(api, "load_graph", lambda *a, **kw: graph)

    def unexpected(*a, **kw):
        raise AssertionError("must not render after a failed transformation")

    class BuggyPass:
        def apply(self, graph):
            raise RuntimeError("programming bug")

    monkeypatch.setattr(api, "render_graph", unexpected)
    with pytest.raises(RuntimeError, match="programming bug"):
        visualize("model.xml", "output.svg", passes=[BuggyPass()])


def test_invalid_pass_configuration_is_rejected_before_reading_model(monkeypatch):
    def unexpected(*a, **kw):
        raise AssertionError("must validate the pass contract before reading weights")

    monkeypatch.setattr(api, "load_graph", unexpected)
    with pytest.raises(TypeError, match="apply"):
        visualize("model.xml", "output.svg", passes=[object()])


@pytest.mark.parametrize("failure_type", ["ModuleNotFoundError", "OSError"])
def test_public_api_and_cli_help_work_without_backend_packages(failure_type):
    code = """
import importlib.abc
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openvino', 'graphviz'}:
            raise ModuleNotFoundError('blocked backend ' + fullname)

sys.meta_path.insert(0, BlockBackends())
from model_graph_viewer import Graph, apply_passes, load_graph, render_graph, visualize
from model_graph_viewer.cli import main
from model_graph_viewer.errors import EnvironmentCheckError
assert apply_passes(Graph()).nodes == {}
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
for operation in [lambda: load_graph('model.xml'), lambda: render_graph(Graph(), 'out.svg')]:
    try:
        operation()
    except EnvironmentCheckError as exc:
        assert 'Cannot import' in str(exc)
    else:
        raise AssertionError('backend import should fail clearly')
assert 'openvino' not in sys.modules
assert 'graphviz' not in sys.modules
"""
    code = code.replace("raise ModuleNotFoundError(", f"raise {failure_type}(")
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
