# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2),),
            ((3, 2, 4),),
            ((2, 3, 4, 5),),
            ((6, 7, 5, 4, 5),),
        ]
    )
    def test_sum_dim_int_default(self, input_shape):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.default(x)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), None, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -3, False),
            ((1, 5, 2, 3), -2, True),
            ((), 0, True),
        ]
    )
    def test_sum_dim_int(self, input_shape, dim, keep_dims):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True),
            ((3, 2, 4), [1], True),
            ((2, 1, 4, 5), None, True),
            ((2, 3, 4, 5), [0, 1, 2, 3], False),
            ((6, 7, 5, 4, 5), [1, 3, 4], False),
            ((6, 7, 5, 4, 5), [-5, -4, -2], False),
        ]
    )
    def test_sum_dim_tuple(self, input_shape, dim, keep_dims):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int32, 0, 5),
            ((2, 3, 4, 5), None, True, torch.int32, -10, 10),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_sum_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True, torch.int32, 0, 5),
            ((3, 2, 4), [1], True, torch.int32, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int32, -10, 10),
            ((2, 3, 4, 5), None, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.bool, 0, 2),
            ((4, 7, 1, 5), None, True, torch.bool, 0, 2),
        ]
    )
    def test_sum_dim_tuple_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ("all", None, False),
            ("empty_dim", [], False),
            ("dim", [1], False),
        ]
    )
    def test_sum_int32_accumulates_in_int64(self, _, dim, keep_dims):
        # PyTorch sums integers in int64. This total is past 2**31, and its partial sums
        # are past 2**24, where float32 rounds.
        class Sum(nn.Module):
            def forward(self, x):
                if dim is None:
                    return torch.ops.aten.sum.default(x)
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.full((2, 4), 2**30 + 1, dtype=torch.int32)]
        self.run_test(Sum(), inputs, rtol=0, atol=0)

    @parameterized.expand(
        [
            ("input", lambda x: torch.ops.aten.sum.default(x)),
            ("slice", lambda x: torch.ops.aten.sum.default(x[:, 1:])),
        ]
    )
    def test_sum_uint8_accumulates_in_int64(self, _, reduce):
        # Each total is past 2**31. After a slice the engine holds uint8 values as int32.
        class Sum(nn.Module):
            def forward(self, x):
                return reduce(x)

        inputs = [torch.full((2, 9_000_001), 255, dtype=torch.uint8)]
        self.run_test(Sum(), inputs, rtol=0, atol=0, use_dynamo_tracer=True)

    def test_sum_bool_accumulates_in_int64(self):
        # The count is int64 in the engine too, so its square does not wrap.
        class Sum(nn.Module):
            def forward(self, x):
                count = torch.ops.aten.sum.default(x)
                return count * count

        inputs = [torch.ones(70000, dtype=torch.bool)]
        self.run_test(Sum(), inputs, rtol=0, atol=0)

    @parameterized.expand(
        [
            ("int32_wraps", torch.int32, torch.int32, 2**30 + 1, 4),
            ("float", torch.float32, torch.float16, 1000.0, 4096),
        ]
    )
    def test_sum_dtype(self, _, dtype, input_dtype, value, size):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.default(x, dtype=dtype)

        inputs = [torch.full((2, size // 2), value, dtype=input_dtype)]
        self.run_test(Sum(), inputs, rtol=0, atol=0)

    @parameterized.expand(
        [
            ("int8_dtype", torch.sum, torch.int32, torch.int8, False, False),
            ("int16_dtype", torch.sum, torch.int32, torch.int16, False, False),
            ("int8_input", torch.sum, torch.int8, None, False, False),
            ("prod_int8_input", torch.prod, torch.int8, None, False, False),
            ("float64_from_int32", torch.sum, torch.int32, torch.float64, True, False),
            ("float64_no_truncate", torch.sum, torch.bool, torch.float64, False, False),
            ("float64_from_bool", torch.sum, torch.bool, torch.float64, True, True),
        ]
    )
    def test_sum_validator(
        self, _, reduce, input_dtype, dtype, truncate_double, in_engine
    ):
        # The validator keeps in PyTorch an int8 input, an int8 or int16 result, and a
        # float64 result from int32 or without truncate_double. A float64 sum of bool runs
        # in TensorRT with truncate_double.
        class Sum(nn.Module):
            def forward(self, x):
                return reduce(x, dtype=dtype)

        model = Sum().eval().cuda()
        x = torch.full((300,), 100, dtype=input_dtype, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            arg_inputs=[x],
            min_block_size=1,
            truncate_double=truncate_double,
        )
        self.assertEqual(
            any("_run_on_acc" in name for name, _ in compiled.named_children()),
            in_engine,
        )
        torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand(
        [
            ("sum_int32", lambda x: torch.sum(x.int())),
            ("prod_float", lambda x: x.float().prod()),
        ]
    )
    def test_reduce_of_cast_int8_runs_in_pytorch(self, _, reduce):
        # TensorRT cannot reduce an int8 engine input after a cast either.
        class Reduce(nn.Module):
            def forward(self, x):
                return reduce(x)

        model = Reduce().eval().cuda()
        x = torch.tensor([[-56, 100, 3], [44, -1, 2]], dtype=torch.int8, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
        )
        names = [name for name, _ in compiled.named_children()]
        self.assertTrue(any("_run_on_gpu" in name for name in names))
        torch.testing.assert_close(compiled(x), model(x))

    def test_reduce_of_data_gathered_by_int8_index_runs_in_tensorrt(self):
        # The int8 value is only the index here, so TensorRT can reduce the data.
        class Reduce(nn.Module):
            def forward(self, x, i):
                return x.index_select(0, i.to(torch.int8).long()).sum()

        model = Reduce().eval().cuda()
        x = torch.randn(5, 3, device="cuda")
        i = torch.tensor([4, 0, 2], dtype=torch.int32, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x, i)),
            arg_inputs=[x, i],
            min_block_size=1,
            require_full_compilation=True,
        )
        torch.testing.assert_close(compiled(x, i), model(x, i))


class TestPrimsSumConverter(DispatchTestCase):
    def test_sum_int32_keeps_dtype(self):
        # prims.sum keeps the input dtype, as the exported graph says, so the total wraps
        # to 0 before the division. Eager prims.sum returns int64, so the expected value
        # is fixed.
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.prims.sum.default(x, [0]) // 2

        x = torch.full((4,), 2**30, dtype=torch.int32, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Sum(), (x,)), arg_inputs=[x], min_block_size=1
        )
        expected = torch.tensor(0, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)

    @parameterized.expand(
        [
            ((3, 2, 4), [1]),
            ((2, 1, 4, 5), [1, 2]),
            ((2, 3, 4, 5), [0, 1, 2, 3]),
            ((6, 7, 5, 4, 5), [1, 3, 4]),
        ]
    )
    def test_sum_dim_sequence(self, input_shape, dim):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.prims.sum.default(x, dim)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ([0], (2, 3), (2, 4), (3, 5)),
            ([1], (2, 3), (2, 4), (3, 5)),
            (
                [
                    2,
                ],
                (2, 2, 4),
                (2, 3, 4),
                (3, 4, 5),
            ),
            ([0, 1], (2, 2, 4), (2, 3, 4), (3, 4, 5)),
        ]
    )
    def test_sum_dynamic_shape(self, dim, min_shape, opt_shape, max_shape):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.prims.sum.default(x, dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sum(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
