import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

empty_ops = [
    (
        "empty_one_dimension",
        [1],
        None,
    ),
    (
        "empty_two_dimension",
        [1, 2],
        None,
    ),
    (
        "empty_three_dimension",
        [2, 3, 4],
        None,
    ),
    (
        "empty_one_dimension_dtype",
        [1],
        torch.float32,
    ),
    (
        "empty_two_dimension_dtype",
        [2, 3],
        torch.float32,
    ),
    (
        "empty_four_dimension_dtype",
        [1, 2, 2, 1],
        torch.float32,
    ),
    (
        "empty_five_dimension_dtype",
        [1, 2, 2, 2, 1],
        torch.float32,
    ),
]


class TestEmptyConverter(DispatchTestCase):
    @parameterized.expand(
        [(empty_op[0], empty_op[1], empty_op[2]) for empty_op in empty_ops]
    )
    def test_empty(self, name, shape_or_input, data_type):
        class TestModule(nn.Module):
            def forward(self, x):
                shape_or_input[0] = x.shape[0]
                return torch.ops.aten.empty.memory_format(
                    shape_or_input,
                    dtype=data_type,
                )

        empty_model = TestModule()

        inputs = [torch.randint(1, 3, shape_or_input, dtype=torch.int32)]
        comparator_shape_dtype_device = (
            lambda x, y, check_dtype: x.shape == y.shape
            and (x.stride() == y.stride())
            and (x.dtype == y.dtype if check_dtype else True)
        )
        expected_ops = []
        if "dtype" in name:
            self.run_test_compare_tensor_attributes_only(
                empty_model,
                inputs,
                expected_ops,
                [(comparator_shape_dtype_device, [True])],
                use_dynamo_tracer=True,
            )
        else:
            self.run_test_compare_tensor_attributes_only(
                empty_model,
                inputs,
                expected_ops,
                [(comparator_shape_dtype_device, [False])],
                use_dynamo_tracer=True,
            )


if __name__ == "__main__":
    run_tests()
