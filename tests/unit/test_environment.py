import subprocess
import sys

import pytest

from model_graph_viewer.cli import main
from model_graph_viewer.environment import check_environment
from model_graph_viewer.errors import EnvironmentCheckError


def test_missing_dot_explains_system_dependency(monkeypatch):
    monkeypatch.setattr("model_graph_viewer.environment.shutil.which", lambda _: None)
    with pytest.raises(EnvironmentCheckError, match="Python graphviz package alone"):
        check_environment()


def test_environment_error_has_nonzero_exit_and_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr("model_graph_viewer.environment.shutil.which", lambda _: None)
    assert main(["--check-environment"]) == 1
    result = capsys.readouterr()
    assert "not found" in result.err
    assert "Traceback" not in result.err


def test_module_help_runs_without_native_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "model_graph_viewer", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check-environment" in result.stdout
