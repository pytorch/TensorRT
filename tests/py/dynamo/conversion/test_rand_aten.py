import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

from .harness import DispatchTestCase

rand_ops = [
    (
        "rand_one_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [1],
    ),
    (
        "rand_two_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [1, 2],
    ),
    (
        "rand_three_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [2, 3, 4],
    ),
    (
        "randn_one_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [1],
    ),
    (
        "randn_two_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2, 3],
    ),
    (
        "randn_three_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2, 3, 4],
    ),
]


rand_perm_ops = [
    (
        "randperm_one_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [1],
    ),
    (
        "randperm_two_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [150],
    ),
    (
        "randperm_three_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [1500],
    ),
]


class TestRandConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                rand_op[0],
                rand_op[1],
                rand_op[2],
            )
            for rand_op in rand_ops
        ]
    )
    def test_rand(self, name, op, shape_or_input):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                shape_or_input[0] = x.shape[0]
                return op(shape_or_input)

        rand_model = TestModule()

        inputs = [torch.randint(1, 3, shape_or_input, dtype=torch.int32)]
        comparator_shape = lambda x, y, check_dtype: x.shape == y.shape and (
            x.dtype == y.dtype if check_dtype else True
        )
        expected_ops = []
        self.run_test_compare_tensor_attributes_only(
            rand_model,
            inputs,
            expected_ops,
            [(comparator_shape, [True])],
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            (
                rand_op[0],
                rand_op[1],
                rand_op[2],
            )
            for rand_op in rand_perm_ops
        ]
    )
    def test_rand(self, name, op, shape_or_input):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                shape_or_input[0] = x.shape[0]
                return op(shape_or_input[0])

        rand_model = TestModule()
        # cannot use self.run_test() since it expects input in form of tensor

        inputs = [torch.randint(1, 3, shape_or_input, dtype=torch.int32)]
        comparator_shape = lambda x, y, check_dtype: x.shape == y.shape and (
            x.dtype == y.dtype if check_dtype else True
        )
        expected_ops = []
        # TRT-TRT returns int32  while torch returns int64
        self.run_test_compare_tensor_attributes_only(
            rand_model,
            inputs,
            expected_ops,
            [(comparator_shape, [False])],
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
