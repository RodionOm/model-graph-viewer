"""Port-aware graph data with no model-framework or rendering dependencies."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from model_graph_viewer.errors import GraphValidationError


def _nonnegative_integer(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise GraphValidationError(f"{label} must be a nonnegative integer, got {value!r}.")


@dataclass(frozen=True, slots=True)
class Dimension:
    """Inclusive dimension bounds; None means no upper bound."""

    lower: int = 0
    upper: int | None = None

    def __post_init__(self) -> None:
        _nonnegative_integer(self.lower, "Dimension lower bound")
        if self.upper is not None:
            _nonnegative_integer(self.upper, "Dimension upper bound")
            if self.upper < self.lower:
                raise GraphValidationError(
                    "Dimension upper bound must be at least its lower bound."
                )

    def __str__(self) -> str:
        if self.lower == self.upper:
            return str(self.lower)
        if self.lower == 0 and self.upper is None:
            return "?"
        return f"{self.lower}..{self.upper if self.upper is not None else '?'}"


@dataclass(frozen=True, slots=True)
class TensorShape:
    """None dimensions means dynamic rank; an empty tuple means a scalar."""

    dimensions: tuple[int | Dimension | None, ...] | None

    def __post_init__(self) -> None:
        if self.dimensions is not None:
            object.__setattr__(self, "dimensions", tuple(self.dimensions))
            for dimension in self.dimensions:
                if dimension is not None and not isinstance(dimension, Dimension):
                    _nonnegative_integer(dimension, "Tensor dimension")

    def __str__(self) -> str:
        if self.dimensions is None:
            return "[...]"
        return "[" + ", ".join("?" if d is None else str(d) for d in self.dimensions) + "]"


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Missing shape metadata is distinct from a known dynamic-rank shape."""

    shape: TensorShape | None = None
    dtype: str | None = None

    def __str__(self) -> str:
        shape = str(self.shape) if self.shape is not None else "[unknown]"
        return f"{self.dtype or '?'}{shape}"


def _ports(values: tuple[int, ...], label: str) -> tuple[int, ...]:
    result = tuple(values)
    for value in result:
        _nonnegative_integer(value, label)
    if len(set(result)) != len(result):
        raise GraphValidationError(f"{label} indices must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class Node:
    """Immutable identity and ports, with mutable operation and visual attributes."""

    node_id: str
    name: str
    op_type: str
    input_ports: tuple[int, ...] = ()
    output_ports: tuple[int, ...] = ()
    attributes: dict[str, object] = field(default_factory=dict)
    visualization: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise GraphValidationError("Node ID must be a nonempty string.")
        object.__setattr__(self, "input_ports", _ports(self.input_ports, "Input port"))
        object.__setattr__(self, "output_ports", _ports(self.output_ports, "Output port"))
        # Own the dictionaries, so changing a caller's dictionary does not replace attributes.
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(self, "visualization", dict(self.visualization))


@dataclass(frozen=True, slots=True)
class Edge:
    from_node: str
    from_port: int
    to_node: str
    to_port: int
    tensor: TensorMetadata = field(default_factory=TensorMetadata)

    def __post_init__(self) -> None:
        _nonnegative_integer(self.from_port, "Source port")
        _nonnegative_integer(self.to_port, "Destination port")


EdgeKey: TypeAlias = tuple[str, int]


class Graph:
    """A directed graph with at most one source per destination input port.

    Structural changes go through this class. Connections are indexed by destination
    port, so parallel edges between the same nodes remain distinct. Adjacency indexes
    make local traversal and node removal proportional to the node's degree.
    """

    def __init__(self, nodes: Iterable[Node] = (), edges: Iterable[Edge] = ()) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[EdgeKey, Edge] = {}
        self._incoming: dict[str, dict[EdgeKey, None]] = {}
        self._outgoing: dict[str, dict[EdgeKey, None]] = {}
        self._input_ports: dict[str, frozenset[int]] = {}
        self._output_ports: dict[str, frozenset[int]] = {}
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Read-only structural view; node attribute dictionaries remain editable."""
        return MappingProxyType(self._nodes)

    @property
    def edges(self) -> Mapping[EdgeKey, Edge]:
        """Read-only view keyed by (destination node ID, destination input port)."""
        return MappingProxyType(self._edges)

    def _node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise GraphValidationError(f"Node {node_id!r} does not exist.") from None

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise GraphValidationError(f"Duplicate node ID {node.node_id!r}.")
        self._nodes[node.node_id] = node
        self._incoming[node.node_id] = {}
        self._outgoing[node.node_id] = {}
        self._input_ports[node.node_id] = frozenset(node.input_ports)
        self._output_ports[node.node_id] = frozenset(node.output_ports)

    def _check_endpoints(self, edge: Edge) -> None:
        self._node(edge.from_node)
        self._node(edge.to_node)
        if edge.from_port not in self._output_ports[edge.from_node]:
            raise GraphValidationError(
                f"Node {edge.from_node!r} has no output port {edge.from_port}."
            )
        if edge.to_port not in self._input_ports[edge.to_node]:
            raise GraphValidationError(f"Node {edge.to_node!r} has no input port {edge.to_port}.")

    def add_edge(self, edge: Edge) -> None:
        self._check_endpoints(edge)
        key = (edge.to_node, edge.to_port)
        if key in self._edges:
            raise GraphValidationError(
                f"Input port {edge.to_port} of node {edge.to_node!r} already has a source."
            )
        self._edges[key] = edge
        self._incoming[edge.to_node][key] = None
        self._outgoing[edge.from_node][key] = None

    def incoming(self, node_id: str) -> tuple[Edge, ...]:
        """Snapshot the visible incoming connections in insertion order."""
        self._node(node_id)
        return tuple(self._edges[key] for key in self._incoming[node_id])

    def outgoing(self, node_id: str) -> tuple[Edge, ...]:
        """Snapshot the visible outgoing connections in insertion order."""
        self._node(node_id)
        return tuple(self._edges[key] for key in self._outgoing[node_id])

    def remove_edge(self, to_node: str, to_port: int) -> Edge:
        _nonnegative_integer(to_port, "Destination port")
        key = (to_node, to_port)
        if key not in self._edges:
            raise GraphValidationError(f"No edge enters node {to_node!r} at port {to_port}.")
        edge = self._edges.pop(key)
        del self._incoming[edge.to_node][key]
        del self._outgoing[edge.from_node][key]
        return edge

    def remove_node(self, node_id: str) -> Node:
        node = self._node(node_id)
        # A self-loop belongs to both indexes; remove it only once.
        keys = self._incoming[node_id].keys() | self._outgoing[node_id].keys()
        for to_node, to_port in keys:
            self.remove_edge(to_node, to_port)
        del self._incoming[node_id]
        del self._outgoing[node_id]
        del self._input_ports[node_id]
        del self._output_ports[node_id]
        del self._nodes[node_id]
        return node

    def validate(self) -> None:
        """Check endpoints and all structural indexes in O(V + E + port count)."""
        for index in (self._incoming, self._outgoing, self._input_ports, self._output_ports):
            if index.keys() != self._nodes.keys():
                raise GraphValidationError("Graph index contains missing or unexpected nodes.")
        for node_id, node in self._nodes.items():
            if node.node_id != node_id:
                raise GraphValidationError("Node ID does not match its graph key.")
            if self._input_ports[node_id] != frozenset(node.input_ports):
                raise GraphValidationError("Input port index is inconsistent.")
            if self._output_ports[node_id] != frozenset(node.output_ports):
                raise GraphValidationError("Output port index is inconsistent.")

        expected_incoming: dict[str, set[EdgeKey]] = {node_id: set() for node_id in self._nodes}
        expected_outgoing: dict[str, set[EdgeKey]] = {node_id: set() for node_id in self._nodes}
        for key, edge in self._edges.items():
            self._check_endpoints(edge)
            if key != (edge.to_node, edge.to_port):
                raise GraphValidationError("Edge key does not match its destination port.")
            expected_incoming[edge.to_node].add(key)
            expected_outgoing[edge.from_node].add(key)
        for node_id in self._nodes:
            if expected_incoming[node_id] != self._incoming[node_id].keys():
                raise GraphValidationError("Incoming edge index is inconsistent.")
            if expected_outgoing[node_id] != self._outgoing[node_id].keys():
                raise GraphValidationError("Outgoing edge index is inconsistent.")
