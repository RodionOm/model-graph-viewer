"""Explicit Python factories and per-step configuration, without a plugin registry."""

import importlib
import json
from pathlib import Path

from model_graph_viewer.errors import ConfigurationError
from model_graph_viewer.passes import (
    FanoutReferencesPass,
    HideConstantsPass,
    HighlightPass,
    PassStep,
)

_BUILTINS = {
    "highlight": HighlightPass,
    "hide-constants": HideConstantsPass,
    "fanout-references": FanoutReferencesPass,
}


def component(reference: str, options: dict | None = None) -> object:
    """Instantiate an explicitly selected module:factory with keyword options.

    Importing a factory executes Python code, just like importing it in an API
    script. Only use component references from code you intend to run.
    """
    parts = reference.split(":")
    if len(parts) != 2 or not all(p and all(s.isidentifier() for s in p.split(".")) for p in parts):
        raise ConfigurationError("A component must be specified as module:factory.")
    module_name, attribute = parts
    try:
        factory = importlib.import_module(module_name)
        for part in attribute.split("."):
            factory = getattr(factory, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(f"Cannot resolve component {reference!r}: {exc}") from exc
    return _construct(factory, options or {}, reference)


def _construct(factory, options: dict, name: str) -> object:
    if not callable(factory):
        raise ConfigurationError(f"Component {name!r} is not callable.")
    try:
        return factory(**options)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid options for {name!r}: {exc}") from exc


def configured_passes(path: str | Path) -> list[PassStep]:
    """Read a JSON array of {name, options, enabled} steps, retaining order."""
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read pass configuration {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ConfigurationError("Pass configuration must be a JSON array.")
    steps = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict) or set(record) - {"name", "options", "enabled"}:
            raise ConfigurationError(f"Step {index}: expected name, options and optional enabled.")
        name = record.get("name")
        options = record.get("options", {})
        enabled = record.get("enabled", True)
        if not isinstance(name, str) or not isinstance(options, dict) or type(enabled) is not bool:
            raise ConfigurationError(f"Step {index}: invalid name, options or enabled value.")
        if name in _BUILTINS:
            transformation = _construct(_BUILTINS[name], options, name)
        elif ":" in name:
            transformation = component(name, options)
        else:
            raise ConfigurationError(f"Step {index}: unknown pass {name!r}.")
        try:
            steps.append(PassStep(transformation, enabled))
        except TypeError as exc:
            raise ConfigurationError(f"Step {index}: {exc}") from exc
    return steps
