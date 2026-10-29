# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _execute_in_torch_example:

Run an annotated region with PyTorch eager or Inductor
======================================================

Run from the repository root with CUDA, Torch-TensorRT, and Triton support for
CUDA Inductor installed::

    python examples/dynamo/execute_in_torch.py

The region contains add -> ReLU -> multiply. This script runs the same model
with each supported backend. ``"eager"`` runs the region's PyTorch operations;
``"inductor"`` compiles them on the first call. In both cases, surrounding
addition and subtraction execute in TensorRT.

``backend="eager"`` is the default; ``"inductor"`` is the only other accepted
value. This selects the compiler, not a ``torch.compile`` tuning mode. The
annotation is a no-op when the original model executes eagerly or is exported
outside Torch-TensorRT. Models with Inductor regions cannot currently be saved
through ``torch_tensorrt.save``.

Inductor may fuse compatible operations, but this example does not promise a
particular kernel count or speed improvement. Use static shapes and the same
inference context for warmup and subsequent calls.
"""

import torch

import torch_tensorrt
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


class Model(torch.nn.Module):
    def __init__(self, backend="eager"):
        super().__init__()
        self.backend = backend

    def forward(self, x, bias, scale):
        x = x + 1.0
        with torch_tensorrt.region.execute_in_torch(backend=self.backend):
            x = x + bias
            x = torch.relu(x)
            x = x * scale
        return x - 0.5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")

    torch.manual_seed(0)
    inputs = tuple(torch.randn(4, 128, device="cuda") for _ in range(3))
    other_inputs = tuple(torch.randn_like(value) for value in inputs)
    for backend in ("eager", "inductor"):
        model = Model(backend=backend).eval().cuda()
        compiled = torch_tensorrt.compile(
            model,
            inputs=list(inputs),
            ir="dynamo",
            strict=True,
            offload_module_to_cpu=False,
            min_block_size=1,
        )

        # The first call compiles an Inductor region. Later calls in the same
        # inference context can reuse it with new values of the same shape.
        with torch.inference_mode():
            for values in (inputs, other_inputs):
                torch.testing.assert_close(compiled(*values), model(*values))

        engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
        assert len(engines) == 2, f"Expected two TensorRT engines, got {len(engines)}"
        print(f"TensorRT addition -> PyTorch {backend} region -> TensorRT subtraction")
        print(compiled.graph)
    print("Both eager and inductor regions match PyTorch on both input cases.")


if __name__ == "__main__":
    main()
