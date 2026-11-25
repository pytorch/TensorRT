# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _region_rules_example:

Place submodules in regions with rules
======================================

Run with CUDA and Torch-TensorRT::

    python examples/dynamo/region_rules.py

A rule places every call of a submodule in a region without editing the
model's source. Pass the rules as ``regions={selector: rule}``. A selector is
a module path, as ``model.named_modules()`` prints it, or a module class. When
you compile the ``torch.nn.Module`` itself, a selector can also be the module
object, such as ``model.blocks[0].mlp``.

This script compiles one small model three ways with the same rules:

* the ``torch.nn.Module`` with ``torch_tensorrt.compile``,
* an exported program that was saved and loaded, with
  ``torch_tensorrt.dynamo.compile``,
* ``torch.compile`` with ``backend="torch_tensorrt"``.

The rules keep each ``mlp`` in PyTorch and replace each ``ScaleNorm`` with a
custom operator. ``inputs=[0, "weight"]`` lists the operator's arguments: the
first tensor the module call receives, then the module's ``weight`` parameter.
The counter below verifies that the compiled model calls the operator. The
model's source has no region code, and ordinary eager calls run it unchanged.

Regions need static shapes, so ``torch.compile`` gets ``dynamic=False``. The
example needs no Triton and builds no TensorRT plugin. :ref:`execute_in_torch`
describes the selectors, the rules, and how each compile path treats them.
"""

import os
import tempfile

import torch
import torch_tensorrt
from torch_tensorrt.region import ExecuteInTorch, FusedOp
from torch_tensorrt.runtime import TorchTensorRTModule

custom_op_calls = 0


@torch.library.custom_op("region_rules_example::scale_norm", mutates_args=())
def scale_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global custom_op_calls
    custom_op_calls += 1
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


@scale_norm.register_fake
def scale_norm_fake(x, weight):
    return torch.empty_like(x)


class ScaleNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(width) + 0.5)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class Mlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = torch.nn.Linear(width, 2 * width)
        self.down = torch.nn.Linear(2 * width, width)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = ScaleNorm(width)
        self.mlp = Mlp(width)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class Model(torch.nn.Module):
    def __init__(self, width=32, depth=2):
        super().__init__()
        self.embed = torch.nn.Linear(width, width)
        self.blocks = torch.nn.ModuleList(Block(width) for _ in range(depth))
        self.head = torch.nn.Linear(width, 8)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# "*" stands for one name, so the path selects blocks.0.mlp and blocks.1.mlp.
# The class selects blocks.0.norm and blocks.1.norm.
RULES = {
    "blocks.*.mlp": ExecuteInTorch(),
    ScaleNorm: FusedOp(scale_norm, inputs=[0, "weight"]),
}


def check(compiled, model, inputs):
    """Compare with PyTorch, and count the custom operator's calls."""
    with torch.no_grad():
        for values in inputs:
            expected = model(values)
            before = custom_op_calls
            actual = compiled(values)
            # One call for each ScaleNorm in the model.
            assert custom_op_calls == before + 2, "The custom operator did not run"
            # TensorRT runs linear layers in TF32 by default: allow a small difference.
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def count_engines(compiled):
    return sum(isinstance(m, TorchTensorRTModule) for m in compiled.modules())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")

    torch.manual_seed(0)
    model = Model().eval().cuda()
    x = torch.randn(4, 32, device="cuda")
    inputs = (x, torch.randn_like(x))

    # 1. Compile the module. Torch-TensorRT traces it and applies the rules.
    from_module = torch_tensorrt.compile(
        model,
        inputs=[x],
        ir="dynamo",
        min_block_size=1,
        regions=RULES,
    )
    check(from_module, model, inputs)
    # The embedding, the addition between the blocks, and the last addition
    # with the head each become one TensorRT engine around the regions.
    assert count_engines(from_module) == 3, "Expected three TensorRT engines"
    print("Module: TensorRT -> [operator -> PyTorch mlp -> TensorRT] x 2")
    print(from_module.graph)

    # 2. Compile a program that was exported and saved earlier. The export
    # needs no region code: the rules select the regions at compile time.
    # A program records each module by path and by the name of its class. The
    # class selector works here because this process has loaded ScaleNorm; a
    # path such as "blocks.*.norm" works in any process.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.pt2")
        torch.export.save(torch.export.export(model, (x,)), path)
        program = torch.export.load(path)
    from_program = torch_tensorrt.dynamo.compile(
        program,
        inputs=[x],
        min_block_size=1,
        regions=RULES,
    )
    check(from_program, model, inputs)
    assert count_engines(from_program) == 3, "Expected three TensorRT engines"
    print("Exported program: the same regions as the module")

    # 3. torch.compile takes the rules as a backend option. It compiles on the
    # first call, so that call is where an error in a rule appears.
    # torch_tensorrt.compile(model, ir="torch_compile", regions=RULES) makes
    # the same call and checks the rules right away.
    just_in_time = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=False,
        options={"min_block_size": 1, "regions": RULES},
    )
    check(just_in_time, model, inputs)
    print("torch.compile: the same regions as the module")

    print(
        "Rules placed the same regions on all three compile paths; "
        "outputs match PyTorch."
    )


if __name__ == "__main__":
    main()
