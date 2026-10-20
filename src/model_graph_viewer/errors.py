"""Errors callers can handle without depending on a specific backend."""


class ViewerError(Exception):
    """An expected failure with a message suitable for CLI users."""


class EnvironmentCheckError(ViewerError):
    """A required dependency is missing or cannot run."""
