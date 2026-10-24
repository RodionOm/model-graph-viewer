"""Create deterministic, untrained OpenVINO models for graph inspection."""

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops


def create_linear_model() -> ov.Model:
    """y = x @ [[0.5], [1.0]] + [1.0]; x has shape [1, 2]."""
    x = ops.parameter([1, 2], ov.Type.f32, name="input")
    weights = ops.constant(np.array([[0.5], [1.0]], dtype=np.float32), name="weights")
    bias = ops.constant(np.array([1.0], dtype=np.float32), name="bias")
    product = ops.matmul(x, weights, False, False, name="matmul")
    result = ops.result(ops.add(product, bias, name="add"), name="output")
    return ov.Model([result], [x], "linear_example")


def create_branching_model() -> ov.Model:
    """Split, parallel connections, a shared constant and two model outputs."""
    x = ops.parameter([2, 4], ov.Type.f32, name="input")
    axis = ops.constant(np.int64(1), name="axis")
    split = ops.split(x, axis, 2, name="split")
    shared = ops.constant(np.array([1, 2], dtype=np.float32), name="shared")
    merge = ops.add(split.output(0), split.output(1), name="merge")
    left = ops.add(merge, shared, name="left")
    right = ops.add(split.output(1), shared, name="right")
    return ov.Model(
        [ops.result(left, name="left_output"), ops.result(right, name="right_output")],
        [x],
        "branching_example",
    )


def create_mlp_model() -> ov.Model:
    """A deterministic 4 -> 3 -> 2 MLP with ReLU and fixed fp32 parameters."""
    x = ops.parameter([1, 4], ov.Type.f32, name="input")
    w1 = ops.constant(np.arange(12, dtype=np.float32).reshape(4, 3) / 10, name="weights1")
    b1 = ops.constant(np.array([0.1, 0.2, 0.3], np.float32), name="bias1")
    w2 = ops.constant(np.arange(6, dtype=np.float32).reshape(3, 2) / 10, name="weights2")
    b2 = ops.constant(np.array([0.1, -0.1], np.float32), name="bias2")
    hidden = ops.relu(
        ops.add(ops.matmul(x, w1, False, False, name="dense1"), b1, name="add1"), name="relu"
    )
    output = ops.add(ops.matmul(hidden, w2, False, False, name="dense2"), b2, name="add2")
    return ov.Model([ops.result(output, name="output")], [x], "mlp_example")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in [
        ("linear", create_linear_model()),
        ("branching", create_branching_model()),
        ("mlp", create_mlp_model()),
    ]:
        path = args.output_dir / f"{name}.xml"
        # serialize preserves the declared weights and operation structure.
        ov.serialize(model, path)
        print(f"Created {path} and {path.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
