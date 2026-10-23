"""Create a deterministic, untrained linear model for inspecting the loader."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "linear.xml"
    # serialize preserves the declared fp32 weights and operation structure.
    ov.serialize(create_linear_model(), path)
    print(f"Created {path} and {path.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
