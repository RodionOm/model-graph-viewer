import json

import pytest

from model_graph_viewer.configuration import component, configured_passes
from model_graph_viewer.errors import ConfigurationError


def test_each_step_has_its_own_options_and_enabled_flag(tmp_path):
    path = tmp_path / "passes.json"
    path.write_text(
        json.dumps(
            [
                {"name": "highlight", "options": {"node_names": ["first"]}},
                {"name": "hide-constants", "enabled": False},
                {"name": "highlight", "options": {"node_names": ["second"]}},
            ]
        )
    )
    steps = configured_passes(path)
    assert steps[0].transformation.node_names == {"first"}
    assert not steps[1].enabled
    assert steps[2].transformation.node_names == {"second"}


@pytest.mark.parametrize(
    "data",
    [
        {},
        [1],
        [{"name": "hide-constants", "typo": 1}],
        [{"name": "unknown"}],
        [{"name": "highlight", "options": {"wrong": 1}}],
        [{"name": "hide-constants", "enabled": "false"}],
        [{"name": "hide-constants", "options": []}],
        [{"name": "builtins:dict"}],
    ],
)
def test_invalid_pass_configuration(data, tmp_path):
    path = tmp_path / "passes.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError):
        configured_passes(path)


@pytest.mark.parametrize(
    "reference", ["missing", "missing_package:Factory", "json:missing", "sys:version"]
)
def test_bad_component_reference(reference):
    with pytest.raises(ConfigurationError):
        component(reference)


def test_malformed_and_missing_config_files(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ConfigurationError, match="Cannot read"):
        configured_passes(path)
    path.write_text("not json")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        configured_passes(path)
