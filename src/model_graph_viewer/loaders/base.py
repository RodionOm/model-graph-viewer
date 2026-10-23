"""The interface a new model-format adapter needs to implement."""

from pathlib import Path
from typing import Protocol

from model_graph_viewer.graph import Graph


class ModelLoader(Protocol):
    def load(self, model_path: str | Path) -> Graph:
        """Read a model and return an independent visualization graph."""
        ...
