# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.conversion import convert_module

from .harness import DispatchTestCase


class TestIsNanConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                torch.tensor(
                    [
                        1.23,
                        float("nan"),
                        -4.56,
                        float("inf"),
                        float("-inf"),
                        -100.0,
                        float("nan"),
                        0.13,
                        -0.13,
                        3.14159265,
                    ]
                ),
            ),
        ]
    )
    def test_isnan_float(self, data):
        class isnan(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isnan.default(input)

        inputs = [data]
        self.run_test(
            isnan(),
            inputs,
        )

    def test_isnan_dynamic_shape_float(self):
        class isnan(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isnan.default(input)

        inputs = [
            Input(
                min_shape=(1, 2, 3),
                opt_shape=(3, 2, 3),
                max_shape=(5, 3, 3),
                dtype=torch.float32,
                torch_tensor=torch.tensor(
                    ([[[3.2, float("nan"), 3.1], [float("inf"), 1.1, float("nan")]]]),
                    dtype=torch.float32,
                ).cuda(),
            )
        ]
        self.run_test_with_dynamic_shape(
            isnan(),
            inputs,
            use_example_tensors=False,
        )

    @parameterized.expand(
        [
            (torch.full((2, 2), float("nan"), dtype=torch.float32),),
            (torch.full((3, 10, 5), float("nan"), dtype=torch.float32),),
            (torch.randn((5, 10, 5), dtype=torch.float32),),
        ]
    )
    def test_isnan_dim(self, data):
        class isnan(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isnan.default(input)

        inputs = [data]
        self.run_test(
            isnan(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_isnan_int(self, input_shape, dtype, low, high):
        class isnan(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isnan.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            isnan(),
            inputs,
        )

    def test_isnan_of_a_constant_without_constant_folding(self):
        """isnan of a constant compares the constant with itself, which gives a constant."""

        class IsNanOfBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("c", torch.tensor([1.0, float("nan"), 2.0]))

            def forward(self, x):
                return torch.ops.aten.add.Tensor(
                    x, torch.ops.aten.isnan.default(self.c)
                )

        x = torch.randn(3).cuda()
        module = torch.export.export(IsNanOfBuffer().cuda(), (x,)).module()
        for node in list(module.graph.nodes):
            if node.op == "call_module" and node.target == "_guards_fn":
                module.graph.erase_node(node)
        module.recompile()
        trt_module = convert_module(
            module, [Input.from_tensor(x)], CompilationSettings(min_block_size=1)
        )
        torch.testing.assert_close(trt_module(x), module(x))


if __name__ == "__main__":
    run_tests()
