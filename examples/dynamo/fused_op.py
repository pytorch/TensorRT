# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _fused_op_example:

Replace an annotated region with a PyTorch custom operator
=========================================================

Run with CUDA and Torch-TensorRT::

    python examples/dynamo/fused_op.py

``fused_op`` replaces the captured add -> ReLU -> multiply region with an
existing custom operator. The operator executes through PyTorch between two
TensorRT engines; it does not require a QDP plugin or Triton kernel. Ordinary
eager calls execute the original region body.

Register the operator and its fake implementation before compiling. Its tensor
arguments follow first use inside the region, and its outputs must match the
entire region. The counter below verifies that compiled inference actually
calls the supplied operator. Register the operator again before loading a
saved model in a new process.
"""

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

custom_op_calls = 0


@torch.library.custom_op("region_fused_op_example::bias_relu_scale", mutates_args=())
def bias_relu_scale(
    x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    global custom_op_calls
    custom_op_calls += 1
    return torch.relu(x + bias) * scale


@bias_relu_scale.register_fake
def bias_relu_scale_fake(x, bias, scale):
    return torch.empty_like(x)


class Model(torch.nn.Module):
    def forward(self, x, bias, scale):
        x = x + 1.0
        with torch_tensorrt.region.fused_op(
            torch.ops.region_fused_op_example.bias_relu_scale.default,
            name="bias_relu_scale",
        ):
            x = torch.relu(x + bias) * scale
        return x - 0.5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")

    torch.manual_seed(0)
    inputs = tuple(torch.randn(4, 128, device="cuda") for _ in range(3))
    model = Model().eval().cuda()
    compiled = torch_tensorrt.compile(
        model,
        inputs=list(inputs),
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,
    )

    engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
    assert len(engines) == 2, f"Expected two TensorRT engines, got {len(engines)}"
    records = get_region_records(compiled)
    assert len(records) == 1
    record = records[0]
    assert record.route == "torch" and record.backend == "eager"
    assert record.owner is not None
    owner = (
        compiled if record.owner == "<root>" else compiled.get_submodule(record.owner)
    )
    assert isinstance(owner, torch.fx.GraphModule)
    children = [
        module
        for module in compiled.modules()
        if getattr(module, "_torch_tensorrt_region_id", None) == record.id
    ]
    assert len(children) == 1
    assert any(
        node.op == "call_function"
        and node.target == torch.ops.region_fused_op_example.bias_relu_scale.default
        for node in children[0].graph.nodes
    )

    with torch.inference_mode():
        for values in (inputs, tuple(torch.randn_like(value) for value in inputs)):
            expected = model(*values)
            before = custom_op_calls
            actual = compiled(*values)
            assert custom_op_calls == before + 1, "The custom operator did not run once"
            torch.testing.assert_close(actual, expected)

    print("TensorRT addition -> PyTorch custom operator -> TensorRT subtraction")
    print(compiled.graph)
    print("The supplied custom operator runs in PyTorch; both input cases match.")


if __name__ == "__main__":
    main()
