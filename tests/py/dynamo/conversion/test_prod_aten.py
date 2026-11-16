# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestProdConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2),),
            ((3, 2, 4),),
            ((2, 3, 4, 5),),
            ((6, 7, 5, 4, 5),),
            ((),),
        ]
    )
    def test_prod_dim_int_default(self, input_shape):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            Prod(),
            inputs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -3, False),
            ((1, 5, 2, 3), -2, True),
        ]
    )
    def test_prod_dim_int(self, input_shape, dim, keep_dims):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim, keep_dims)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            Prod(),
            inputs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_prod_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Prod(),
            inputs,
            check_dtype=False,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("all", None, False),
            ("dim", 1, False),
            ("dim_keepdim", 1, True),
        ]
    )
    def test_prod_int32_accumulates_in_int64(self, _, dim, keep_dims):
        # PyTorch multiplies integers in int64: 50001**2 is past 2**31, and 50001**4
        # still fits in int64.
        class Prod(nn.Module):
            def forward(self, x):
                if dim is None:
                    return torch.prod(x)
                return torch.prod(x, dim, keep_dims)

        inputs = [torch.full((2, 2), 50001, dtype=torch.int32)]
        self.run_test(Prod(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    def test_prod_dim_int_dtype(self):
        # 300 * 300 overflows float16 but not the requested float32.
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, 1, dtype=torch.float32)

        inputs = [torch.tensor([[300, 300], [3, 5]], dtype=torch.float16)]
        self.run_test(Prod(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    @parameterized.expand([("input", False), ("slice", True)])
    def test_prod_uint8_accumulates_in_int64(self, _, sliced):
        # 250**4 is past 2**31. After a slice the engine holds uint8 values as int32.
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x[:, 1:] if sliced else x, 1)

        inputs = [
            torch.tensor([[1, 250, 250, 250, 250], [1, 3, 5, 7, 1]], dtype=torch.uint8)
        ]
        self.run_test(Prod(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    def test_prod_bool(self):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, 1)

        inputs = [torch.tensor([[True, True], [True, False]])]
        self.run_test(Prod(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    def test_prod_int32_is_exact_past_2_24(self):
        # 4097 * 4097 fits in int32 but is past 2**24, where float32 rounds.
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dtype=torch.int32)

        inputs = [torch.tensor([4097, 4097], dtype=torch.int32)]
        self.run_test(Prod(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (0, (2, 3), (2, 4), (3, 5)),
            (1, (2, 3), (2, 4), (3, 5)),
            (2, (2, 2, 4), (2, 3, 4), (3, 4, 5)),
            (-1, (2, 2, 4), (2, 3, 4), (3, 4, 5)),
        ]
    )
    def test_prod_dynamic_shape(self, dim, min_shape, opt_shape, max_shape):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Prod(),
            input_specs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
