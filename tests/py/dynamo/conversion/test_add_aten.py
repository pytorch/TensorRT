# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAddConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_add_tensor(self, _, shape):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_add_tensor_alpha(self, _, shape, alpha):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val, alpha=alpha)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_add_scalar(self, _, shape, scalar):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, 1.0),
            ("3d", (2, 1, 2), 2, 2),
        ]
    )
    def test_add_scalar_alpha(self, _, shape, scalar, alpha):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar, alpha=alpha)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    def test_add_alpha_names_its_layers_apart(self):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.add(lhs_val, rhs_val, alpha=3)

        inputs = (torch.randn(2, 3).cuda(), torch.randn(2, 3).cuda())
        # A timing cache of its own: TensorRT logs an error when it reads the shared
        # default one while another test process writes it.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
            with self.assertNoLogs(
                "torch_tensorrt [TensorRT Conversion Context]", level="ERROR"
            ):
                torch_tensorrt.dynamo.compile(
                    torch.export.export(add().cuda().eval(), inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    timing_cache_path=os.path.join(cache_dir, "timing_cache.bin"),
                )

    def test_add_int8_buffer(self):
        # TensorRT reads an INT8 constant as quantized data, so a plain int8 weight
        # must not become one.
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
                )

            def forward(self, x):
                return torch.ops.aten.add.Tensor(self.offsets, x)

        inputs = [torch.randn(16, 16)]
        self.run_test(
            add(),
            inputs,
            rtol=0,
            atol=0,
        )

    def test_add_int8_buffer_computes_in_int32(self):
        # Casting the widened constant back to int8 inside the network sent the TensorRT
        # builder into a very long search on a large model, so the sum stays int32 and an
        # int8 sum that overflows does not wrap around as in PyTorch.
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.tensor([100, -100, 120, 5], dtype=torch.int8)
                )

            def forward(self, x):
                total = torch.ops.aten.add.Tensor(self.offsets, x.to(torch.int8))
                return total.to(torch.float32)

        model = add().cuda().eval()
        inputs = (torch.tensor([100.0, -100.0, 120.0, 5.0]).cuda(),)
        with self.assertLogs(
            "torch_tensorrt.dynamo.conversion.impl.elementwise.base", level="WARNING"
        ) as logs:
            trt_model = torch_tensorrt.dynamo.compile(
                torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
            )
        self.assertTrue(any("does not wrap around" in line for line in logs.output))
        torch.testing.assert_close(
            trt_model(*inputs).cpu(),
            torch.tensor([200.0, -200.0, 240.0, 10.0]),
            rtol=0,
            atol=0,
        )
        # PyTorch wraps around in int8; the engine does not, as the warning says.
        torch.testing.assert_close(
            model(*inputs).cpu(), torch.tensor([-56.0, 56.0, -16.0, 10.0])
        )

    def test_add_int8_buffer_int8_output_wraps(self):
        # An int8 graph output is cast back from int32 once, at the engine output, so it
        # wraps like PyTorch.
        class add(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offsets", torch.tensor([100, -100, 120, 5], dtype=torch.int8)
                )

            def forward(self, x):
                return torch.ops.aten.add.Tensor(self.offsets, x.to(torch.int8))

        model = add().cuda().eval()
        inputs = (torch.tensor([100.0, -100.0, 120.0, 5.0]).cuda(),)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
        )
        out = trt_model(*inputs)
        self.assertEqual(out.dtype, torch.int8)
        torch.testing.assert_close(
            out.cpu(), torch.tensor([-56, 56, -16, 10], dtype=torch.int8)
        )

    def test_add_python_int_to_int8_warns(self):
        class add(nn.Module):
            def forward(self, x):
                return (x.to(torch.int8) + 100).to(torch.float32)

        model = add().cuda().eval()
        inputs = (torch.tensor([100.0, -100.0, 120.0, 5.0]).cuda(),)
        with self.assertLogs(
            "torch_tensorrt.dynamo.conversion.impl.elementwise.base", level="WARNING"
        ) as logs:
            torch_tensorrt.dynamo.compile(
                torch.export.export(model, inputs), arg_inputs=inputs, min_block_size=1
            )
        self.assertTrue(any("does not wrap around" in line for line in logs.output))


if __name__ == "__main__":
    run_tests()
