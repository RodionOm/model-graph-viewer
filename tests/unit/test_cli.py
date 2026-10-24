from pathlib import Path

import pytest

from model_graph_viewer import cli
from model_graph_viewer.errors import ModelLoadError
from model_graph_viewer.passes import HideConstantsPass, HighlightPass


@pytest.mark.parametrize(
    "options, expected",
    [
        (["--model", "model.xml"], "both --model and --output"),
        (["--output", "out.svg"], "both --model and --output"),
        (["--weights", "model.bin"], "both --model and --output"),
        (["--model", "m.xml", "--output", "out.png"], "must end with .svg"),
        (["--passes", "typo"], "unknown pass"),
        (["--passes", "hide-constants,"], "unknown pass ''"),
        (["--passes", ""], "unknown pass ''"),
        (["--passes", "highlight"], "at least one --highlight-node"),
        (["--highlight-node", "name"], "requires highlight"),
        (["--passes", "hide-constants", "--highlight-node", "name"], "requires highlight"),
        (["--timeout", "nan"], "finite positive"),
        (["--timeout", "inf"], "finite positive"),
        (["--timeout", "0"], "finite positive"),
        (["--timeout", "-5"], "finite positive"),
        (["--timeout", "abc"], "positive number"),
        (["--layout", "unknown"], "invalid choice"),
        (["--max-label-length", "7"], "integer of at least 8"),
        (["--max-label-length", "8.5"], "integer of at least 8"),
        (["--check-environment", "--timeout", "5"], "cannot be combined"),
        (["--mod", "m.xml"], "unrecognized arguments"),
    ],
)
def test_cli_rejects_invalid_arguments_before_api_calls(monkeypatch, capsys, options, expected):
    def unexpected(*a, **kw):
        raise AssertionError("invalid arguments must not invoke the API")

    monkeypatch.setattr(cli, "visualize", unexpected)
    monkeypatch.setattr(cli, "check_environment", unexpected)
    # Partial-path cases intentionally omit the other required argument.
    if not any(
        name in options for name in ("--model", "--output", "--weights", "--check-environment")
    ):
        options = ["--model", "m.xml", "--output", "out.svg", *options]
    with pytest.raises(SystemExit) as exc:
        cli.main(options)
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_cli_preserves_pass_order_repetitions_and_literal_names(monkeypatch, capsys):
    def visualize(model_path, output_path, **kwargs):
        assert model_path == Path("model space.xml")
        assert output_path == Path("граф output.svg")
        assert kwargs["weights_path"] == Path("weights space.bin")
        steps = kwargs["passes"]
        assert [type(step) for step in steps] == [
            HideConstantsPass,
            HighlightPass,
            HideConstantsPass,
            HighlightPass,
        ]
        assert steps[1].node_names == frozenset({"узел, с пробелом", " another name "})
        assert steps[3].node_names == steps[1].node_names
        assert kwargs["max_label_length"] == 80
        assert kwargs["timeout"] == 2.5
        return output_path

    monkeypatch.setattr(cli, "visualize", visualize)
    assert (
        cli.main(
            [
                "--model",
                "model space.xml",
                "--output",
                "граф output.svg",
                "--weights",
                "weights space.bin",
                "--passes",
                "hide-constants, highlight",
                "--passes",
                "hide-constants,highlight",
                "--highlight-node",
                "узел, с пробелом",
                "--highlight-node",
                " another name ",
                "--max-label-length",
                "80",
                "--timeout",
                "2.5",
            ]
        )
        == 0
    )
    assert "Created SVG" in capsys.readouterr().out


def test_cli_default_has_no_implicit_passes(monkeypatch):
    def visualize(model_path, output_path, **kwargs):
        assert kwargs["passes"] == []
        assert kwargs["weights_path"] is None
        assert kwargs["timeout"] is None
        assert kwargs["max_label_length"] == 120
        return output_path

    monkeypatch.setattr(cli, "visualize", visualize)
    assert cli.main(["--model", "m.xml", "--output", "out.svg"]) == 0


def test_expected_cli_error_is_short_and_has_no_traceback(monkeypatch, capsys):
    def fail(*a, **kw):
        raise ModelLoadError("Invalid model\n" + "backend details " * 300)

    monkeypatch.setattr(cli, "visualize", fail)
    assert cli.main(["--model", "m.xml", "--output", "out.svg"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "Invalid model" in output.err
    assert "Traceback" not in output.err
    assert len(output.err.splitlines()) == 1
    assert len(output.err) < 1100


def test_cli_does_not_hide_unexpected_programming_errors(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(cli, "visualize", fail)
    with pytest.raises(RuntimeError, match="programming bug"):
        cli.main(["--model", "m.xml", "--output", "out.svg"])


def test_cli_no_arguments_prints_help(capsys):
    assert cli.main([]) == 0
    assert "--model" in capsys.readouterr().out
