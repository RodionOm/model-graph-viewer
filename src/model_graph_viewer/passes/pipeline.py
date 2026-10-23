"""Run configured passes in order without copying the visualization graph."""

from collections.abc import Iterable
from dataclasses import dataclass

from model_graph_viewer.errors import GraphValidationError
from model_graph_viewer.graph import Graph
from model_graph_viewer.passes.base import TransformationPass


@dataclass(frozen=True, slots=True)
class PassStep:
    """Attach an enabled flag to any pass without changing its implementation."""

    transformation: TransformationPass
    enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("PassStep.enabled must be a boolean.")
        if not callable(getattr(self.transformation, "apply", None)):
            raise TypeError("A transformation must provide an apply(graph) method.")


class Pipeline:
    """Apply passes in sequence, validating before and after each enabled pass.

    Plain passes are enabled by default. PassStep allows individual passes to be
    disabled. Failures stop execution; completed changes are not rolled back.
    """

    def __init__(self, passes: Iterable[TransformationPass | PassStep] = ()) -> None:
        self.steps = tuple(
            step if isinstance(step, PassStep) else PassStep(step) for step in passes
        )

    def apply(self, graph: Graph) -> Graph:
        """Return the same graph after applying enabled passes in configured order."""
        graph.validate()
        for index, step in enumerate(self.steps, start=1):
            if not step.enabled:
                continue
            name = type(step.transformation).__name__
            try:
                result = step.transformation.apply(graph)
                if result is not None:
                    raise TypeError(
                        f"Pass {index} ({name}) must modify the graph in place and return None."
                    )
                graph.validate()
            except GraphValidationError as exc:
                raise GraphValidationError(f"Pass {index} ({name}) failed: {exc}") from exc
        return graph
