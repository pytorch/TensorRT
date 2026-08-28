import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestArangeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (0, 5, 1),
            (1, 5, 2),
            (3, 5, 3),
            (5, 0, -1),
            (5, 1, -2),
            (5, 3, -3),
            (5, -2, -1),
            (-5, -2, 2),
            (-5, -3, 1),
            (-2, -5, -1),
            (1.2, 5, 1.3),
            (1.2, 5.0, 1.3),
            (1, 5.0, 1.3),
            (-1.2, -5.0, -1.3),
            (-5.0, -1.2, 1.3),
            (-5, 1.2, 1.3),
            (-5.0, 1, 1.3),
        ]
    )
    def test_arange(self, start, end, step):
        class Arange(nn.Module):
            def forward(self, x):
                return torch.ops.aten.arange.start_step(start, end, step)

        inputs = [torch.randn(1, 1)]
        self.run_test(
            Arange(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_arange_dynamic_int32(self):
        class Arange(nn.Module):
            def forward(self, end_tensor):
                return torch.ops.aten.arange.start_step(0, end_tensor, 1)

        pyt_input = 7
        inputs = [
            torch_tensorrt.Input(
                min_shape=(5,),
                opt_shape=(7,),
                max_shape=(10,),
                dtype=torch.int32,
                torch_tensor=torch.tensor(pyt_input, dtype=torch.int32).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=[pyt_input],
            use_dynamo_tracer=False,
        )

    def test_arange_dynamic_int64(self):
        class Arange(nn.Module):
            def forward(self, end_tensor):
                return torch.ops.aten.arange.start_step(0, end_tensor, 1)

        pyt_input = 7
        inputs = [
            torch_tensorrt.Input(
                min_shape=(5,),
                opt_shape=(7,),
                max_shape=(10,),
                dtype=torch.int64,
                torch_tensor=torch.tensor(pyt_input, dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=[pyt_input],
            use_dynamo_tracer=False,
        )

    @parameterized.expand([("int32", torch.int32), ("int64", torch.int64)])
    def test_arange_dynamic_start(self, _, dtype):
        """A dynamic `start` reaches the Fill layer as an ITensor.

        LINSPACE requires `alpha` (start) and `beta` (step) to share a dtype. A dynamic
        `start` keeps the dtype of its incoming ITensor, while a literal `step` is
        materialized as a constant of the sequence dtype, so the two can differ.

        Three cases with a dynamic bound are covered here: an int32 `start`, an int64
        `start`, and both bounds dynamic. Of those, only the int64 `start` works today,
        because the sequence dtype for integer operands is already int64 and matches.
        """

        class Arange(nn.Module):
            def forward(self, start_tensor):
                return torch.ops.aten.arange.start_step(start_tensor, 10, 1)

        pyt_input = 2
        inputs = [
            torch_tensorrt.Input(
                min_shape=(0,),
                opt_shape=(2,),
                max_shape=(5,),
                dtype=dtype,
                torch_tensor=torch.tensor(pyt_input, dtype=dtype).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=[pyt_input],
            use_dynamo_tracer=False,
        )

    def test_arange_dynamic_start_and_end(self):
        """Both bounds dynamic, so neither is a constant carrying the sequence dtype."""

        class Arange(nn.Module):
            def forward(self, start_tensor, end_tensor):
                return torch.ops.aten.arange.start_step(start_tensor, end_tensor, 1)

        pyt_inputs = [2, 9]
        inputs = [
            torch_tensorrt.Input(
                min_shape=(0,),
                opt_shape=(2,),
                max_shape=(5,),
                dtype=torch.int32,
                torch_tensor=torch.tensor(pyt_inputs[0], dtype=torch.int32).cuda(),
                is_shape_tensor=True,
            ),
            torch_tensorrt.Input(
                min_shape=(6,),
                opt_shape=(9,),
                max_shape=(12,),
                dtype=torch.int64,
                torch_tensor=torch.tensor(pyt_inputs[1], dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Arange(),
            inputs,
            use_example_tensors=False,
            check_dtype=False,
            pyt_inputs=pyt_inputs,
            use_dynamo_tracer=False,
        )


if __name__ == "__main__":
    run_tests()
