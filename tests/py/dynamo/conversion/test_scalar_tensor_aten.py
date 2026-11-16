# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestScalarTensorConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (-2.00001,),
            (-1.3,),
            (-0.0,),
            (1.0,),
            (2.99,),
        ]
    )
    def test_scalar_tensor_float(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    @parameterized.expand(
        [
            (-9999,),
            (-1,),
            (0,),
            (2,),
            (99999,),
        ]
    )
    def test_scalar_tensor_int(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_scalar_tensor_bool(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    @parameterized.expand(
        [
            (-9999, torch.int),
            (-2.00001, torch.float),
            (-1, torch.float),
            (0, torch.int),
            (-0.0, torch.float),
            (1.0, torch.int),
            (2.99, torch.float),
            (9999999, None),
            (9999999.99999, None),
            (True, torch.bool),
        ]
    )
    def test_scalar_tensor_dtype(self, scalar, dtype):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar, dtype=dtype)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    # The tests above use torch.fx.symbolic_trace, which folds a scalar_tensor call with
    # constant arguments into a constant. torch.export keeps the call for the converter.
    @parameterized.expand(
        [("float", 2.5, None), ("int", 7, torch.int32), ("bool", True, torch.bool)]
    )
    def test_scalar_tensor_is_rank_zero(self, _, scalar, dtype):
        class ScalarTensor(nn.Module):
            def forward(self, x):
                return torch.ops.aten.scalar_tensor.default(scalar, dtype=dtype)

        self.run_test(ScalarTensor(), [torch.randn(2)], use_dynamo_tracer=True)

    def test_scalar_tensor_defaults_to_float(self):
        # Without a dtype, PyTorch returns float32: fmod(7, 2.5) is 2, not int32's 1.
        class ScalarFmod(nn.Module):
            def forward(self, x):
                scalar = torch.ops.aten.scalar_tensor.default(7, device=x.device)
                return torch.ops.aten.fmod.Scalar(scalar, 2.5) + x

        self.run_test(ScalarFmod(), [torch.randn(2)], use_dynamo_tracer=True)

    @parameterized.expand(
        [
            ("rank", torch.float32, False),
            ("default_dtype", None, True),
            ("uint8", torch.uint8, False),
        ]
    )
    def test_scalar_tensor_from_sym_size(self, _, dtype, use_fmod):
        # A SymInt arrives as the int32, rank-1 output of sym_size. The cat checks the
        # rank; fmod checks the default float32 dtype (in int32, 2.5 truncates to 2);
        # uint8 must stay int32 inside the engine, as TensorRT allows uint8 only on I/O.
        class SizeTensor(nn.Module):
            def forward(self, x):
                size = torch.ops.aten.scalar_tensor.default(
                    x.shape[0], dtype=dtype, device=x.device
                )
                if use_fmod:
                    return torch.ops.aten.fmod.Scalar(size, 2.5) + x
                return torch.ops.aten.cat.default(
                    [torch.ops.aten.unsqueeze.default(size, 0), x[0]], 0
                )

        input_specs = [
            Input(
                min_shape=(1, 4),
                opt_shape=(3, 4),
                max_shape=(8, 4),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(
            SizeTensor(), input_specs, use_dynamo_tracer=True
        )


if __name__ == "__main__":
    run_tests()
