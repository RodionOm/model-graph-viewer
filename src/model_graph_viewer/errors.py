"""Errors callers can handle without depending on a specific backend."""


class ViewerError(Exception):
    """An expected failure with a message suitable for CLI users."""


class EnvironmentCheckError(ViewerError):
    """A required dependency is missing or cannot run."""


class GraphValidationError(ViewerError):
    """A graph operation would violate its structural invariants."""


class ModelLoadError(ViewerError):
    """A model file cannot be read or converted to the intermediate graph."""


class RenderError(ViewerError):
    """A graph cannot be rendered or its output cannot be saved."""
