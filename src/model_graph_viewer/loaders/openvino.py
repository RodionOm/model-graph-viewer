"""Read OpenVINO IR without compiling, transforming, or copying model weights."""

from pathlib import Path

import openvino as ov

from model_graph_viewer.errors import ModelLoadError
from model_graph_viewer.graph import Dimension, Edge, Graph, Node, TensorMetadata, TensorShape

_DTYPE_NAMES = {"f16": "fp16", "f32": "fp32", "f64": "fp64"}


def _shape(shape: ov.PartialShape) -> TensorShape:
    if shape.rank.is_dynamic:
        return TensorShape(None)
    dimensions: list[int | Dimension | None] = []
    for dimension in shape:
        if dimension.is_static:
            dimensions.append(dimension.get_length())
        else:
            lower = dimension.get_min_length()
            upper = dimension.get_max_length()
            if lower == 0 and upper == -1:
                dimensions.append(None)
            else:
                dimensions.append(Dimension(lower, None if upper == -1 else upper))
    return TensorShape(tuple(dimensions))


def _tensor(output: ov.Output) -> TensorMetadata:
    element_type = output.get_element_type()
    name = element_type.get_type_name()
    dtype = None if element_type.is_dynamic() else _DTYPE_NAMES.get(name, name)
    return TensorMetadata(_shape(output.get_partial_shape()), dtype)


def _attribute(value: object) -> object:
    """Detach operation attributes without reading arbitrary arrays or tensors."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_attribute(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _attribute(item) for key, item in value.items()}
    if isinstance(value, ov.PartialShape):
        return str(_shape(value))
    if isinstance(value, (ov.Dimension, ov.Type)):
        return str(value)
    # Never call str(), tolist(), or data accessors on an unknown tensor-like value.
    return f"<unsupported attribute type: {type(value).__name__}>"


class OpenVINOLoader:
    """Load IR files or parse an existing openvino.Model into ordinary Python data.

    Node IDs are the string indices of get_ordered_ops(), independent of friendly
    names. They are repeatable for the same IR and runtime, not persistent IDs
    across model edits or OpenVINO versions. Only the outer graph is represented.
    """

    def read_model(
        self, model_path: str | Path, weights_path: str | Path | None = None
    ) -> ov.Model:
        """Read XML and its companion BIN; a weightless IR may omit the BIN."""
        path = Path(model_path)
        if path.suffix.lower() != ".xml":
            raise ModelLoadError(f"Unsupported model format: {path}. Expected OpenVINO IR (.xml).")
        if not path.is_file():
            raise ModelLoadError(f"Model file does not exist or is not a regular file: {path}")
        weights = Path(weights_path) if weights_path is not None else path.with_suffix(".bin")
        if weights_path is not None and not weights.is_file():
            raise ModelLoadError(f"Weights file does not exist or is not a regular file: {weights}")
        try:
            # Let OpenVINO decide whether weights are required by this particular IR.
            return ov.Core().read_model(str(path), str(weights) if weights.is_file() else "")
        except (RuntimeError, OSError) as exc:
            hint = "" if weights.is_file() else f" Companion weights file was not found: {weights}."
            raise ModelLoadError(f"Cannot read OpenVINO IR {path}.{hint}\n{exc}") from exc

    def parse(self, model: ov.Model) -> Graph:
        """Build a detached graph; do not retain operations or tensor buffers."""
        if not isinstance(model, ov.Model):
            raise ModelLoadError("Expected an openvino.Model to parse.")
        operations = model.get_ordered_ops()
        node_ids = {operation: str(index) for index, operation in enumerate(operations)}
        graph = Graph()
        tensors: dict[tuple[str, int], TensorMetadata] = {}

        for operation in operations:
            node_id = node_ids[operation]
            op_type = operation.get_type_name()
            for output in operation.outputs():
                tensors[(node_id, output.get_index())] = _tensor(output)
            # get_attributes on Constant can expose its value. Never call it here.
            if op_type == "Constant":
                metadata = tensors[(node_id, 0)]
                attributes = {"dtype": metadata.dtype, "shape": str(metadata.shape)}
            else:
                attributes = {
                    key: _attribute(value) for key, value in operation.get_attributes().items()
                }
                if op_type == "Parameter":
                    # get_attributes may flatten dimension intervals to -1.
                    attributes["shape"] = str(tensors[(node_id, 0)].shape)
            node = Node(
                node_id=node_id,
                name=operation.get_friendly_name(),
                op_type=op_type,
                input_ports=tuple(port.get_index() for port in operation.inputs()),
                output_ports=tuple(port.get_index() for port in operation.outputs()),
                attributes=attributes,
            )
            if op_type in {"If", "Loop", "TensorIterator"}:
                node.visualization["note"] = "Nested bodies are not expanded."
            graph.add_node(node)

        for operation in operations:
            for port in operation.inputs():
                source = port.get_source_output()
                source_id = node_ids[source.get_node()]
                graph.add_edge(
                    Edge(
                        from_node=source_id,
                        from_port=source.get_index(),
                        to_node=node_ids[operation],
                        to_port=port.get_index(),
                        tensor=tensors[(source_id, source.get_index())],
                    )
                )
        graph.validate()
        return graph

    def load(self, model_path: str | Path, weights_path: str | Path | None = None) -> Graph:
        """Read an IR file and construct its visualization graph."""
        return self.parse(self.read_model(model_path, weights_path))
