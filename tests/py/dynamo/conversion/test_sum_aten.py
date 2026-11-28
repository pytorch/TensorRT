# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import patch

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
            ("bool_dtype", torch.sum, torch.int32, torch.bool, False, False),
            ("float64_from_int32", torch.sum, torch.int32, torch.float64, True, True),
            ("float64_from_bool", torch.sum, torch.bool, torch.float64, True, True),
            ("float64_from_float", torch.sum, torch.float32, torch.float64, True, True),
        ]
    )
    def test_sum_validator(
        self, _, reduce, input_dtype, dtype, truncate_double, in_engine
    ):
        # The validator keeps a bool result in PyTorch. A float64 result runs in
        # TensorRT, also with truncate_double.
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
            ("sum_float64", torch.sum, torch.float32, torch.float64, [3, -2, 5]),
            ("prod_float64", torch.prod, torch.float32, torch.float64, [3, -2, 5]),
            # float16 would overflow past 65504.
            (
                "sum_float64_of_half",
                torch.sum,
                torch.float16,
                torch.float64,
                [1e3] * 70,
            ),
            # PyTorch casts the input to the dtype first, so 1.7 counts as 1.
            ("sum_int16_of_float", torch.sum, torch.float32, torch.int16, [0.5, 1.7]),
            # An int64 product would wrap 2**64 to 0.
            (
                "prod_float64_of_int32",
                torch.prod,
                torch.int32,
                torch.float64,
                [2**16] * 4,
            ),
        ]
    )
    def test_reduce_read_by_next_op_stays_in_tensorrt(
        self, _, reduce, input_dtype, dtype, values
    ):
        # No engine takes such a result as an input, so the reduction runs in a dtype the
        # engine holds and stays in the engine of the op that reads it.
        class Reduce(nn.Module):
            def forward(self, x):
                return reduce(x, -1, dtype=dtype).float() * 2

        model = Reduce().eval().cuda()
        x = torch.tensor([values, values[::-1]], dtype=input_dtype, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
        )
        names = [name for name, _ in compiled.named_children()]
        self.assertFalse(any("_run_on_gpu" in name for name in names))
        torch.testing.assert_close(compiled(x), model(x))

    @parameterized.expand(
        [
            (
                "to_uint8",
                [[1.0, 2.0, 3.0], [10.0, 10.0, 0.0]],
                lambda s: s.to(torch.uint8),
            ),
            (
                "avg_pool_of_int64",
                [[[[[1, 0], [2, 0]], [[1, 1], [2, 0]]]]],
                lambda s: torch.nn.functional.avg_pool2d(s, 2).float(),
            ),
            ("int_total_past_2_24", [[2**24 + 1] * 3], lambda s: s.float()),
        ]
    )
    def test_sum_float64_read_by_next_op(self, _, values, consume):
        # The cast to uint8 of a float64 runs in PyTorch, so the sum becomes an engine
        # output. An integer total is a float, so the pooling averages it. 2**24 + 1
        # rounds in float32, so only a total reduced in int64 is exact.
        class Sum(nn.Module):
            def forward(self, x):
                return consume(x.sum(-1, dtype=torch.float64))

        model = Sum().eval().cuda()
        x = torch.tensor(values, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
        )
        torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)

    def test_sum_int8_read_in_pytorch_on_any_tensorrt(self):
        # The cast to uint8 of an int8 value runs in PyTorch, so the sum becomes an engine
        # output when the interpreter validates it again. No TensorRT version refuses it
        # then, and the cast to int8 at the engine output wraps 300 to 44.
        class Sum(nn.Module):
            def forward(self, x):
                return x.sum(-1, dtype=torch.int8).to(torch.uint8)

        model = Sum().eval().cuda()
        x = torch.tensor([[100, 100, 100], [50, 30, 20]], device="cuda")
        with patch(
            "torch_tensorrt.dynamo.conversion.aten_ops_converters."
            "is_tensorrt_version_supported",
            return_value=False,
        ):
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
            )
        names = [name for name, _ in compiled.named_children()]
        self.assertTrue(any("_run_on_acc" in name for name in names))
        torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)


class TestPrimsSumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("int8", torch.int8, [5.0, 7.0]),
            ("bool", torch.bool, [2.0, 2.0]),
        ]
    )
    def test_sum_output_dtype(self, _, output_dtype, expected):
        # The engine casts the sum to output_dtype, so 260 wraps to 4. Eager prims.sum
        # cannot run with an output_dtype, so the expected values are fixed.
        class Sum(nn.Module):
            def forward(self, x):
                total = torch.ops.prims.sum.default(x, [1], output_dtype=output_dtype)
                return total.float() + 1

        x = torch.tensor([[100, 100, 60], [1, 2, 3]], dtype=torch.int32, device="cuda")
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Sum(), (x,)), arg_inputs=[x], min_block_size=1
        )
        torch.testing.assert_close(
            compiled(x), torch.tensor(expected, device="cuda"), rtol=0, atol=0
        )

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
