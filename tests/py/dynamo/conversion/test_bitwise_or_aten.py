import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestBitwiseOrConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 3), (2, 3)),
            ("3d", (5, 3, 2), (5, 3, 2)),
            ("3d_broadcast", (2, 3), (2, 1, 3)),
            ("4d_broadcast_1", (2, 3), (1, 2, 1, 3)),
            ("4d_broadcast_2", (2, 3), (2, 2, 2, 3)),
        ]
    )
    def test_bitwise_or_tensor(self, _, lhs_shape, rhs_shape):
        class bitwise_or(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_or.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 2, lhs_shape, dtype=bool),
            torch.randint(0, 2, rhs_shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d-2d", (2, 3), (3, 3), (5, 3), (2, 3), (3, 3), (5, 3)),
            ("3d-3d", (2, 2, 2), (2, 3, 2), (2, 4, 2), (1, 2, 2), (1, 3, 2), (1, 4, 2)),
        ]
    )
    def test_bitwise_or_tensor_dynamic_shape(
        self,
        _,
        lhs_min_shape,
        lhs_opt_shape,
        lhs_max_shape,
        rhs_min_shape,
        rhs_opt_shape,
        rhs_max_shape,
    ):
        class bitwise_or(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_or.Tensor(lhs_val, rhs_val)

        inputs = [
            Input(
                dtype=torch.bool,
                min_shape=lhs_min_shape,
                opt_shape=lhs_opt_shape,
                max_shape=lhs_max_shape,
                torch_tensor=torch.randint(0, 2, lhs_opt_shape, dtype=bool),
            ),
            Input(
                dtype=torch.bool,
                min_shape=rhs_min_shape,
                opt_shape=rhs_opt_shape,
                max_shape=rhs_max_shape,
                torch_tensor=torch.randint(0, 2, rhs_opt_shape, dtype=bool),
            ),
        ]
        self.run_test_with_dynamic_shape(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )

    # Only False is here; `x | True` is handled by TestBitwiseOrTrueScalar
    # below, because the validator keeps it out of TensorRT.
    @parameterized.expand(
        [
            ("2d", (5, 3), False),
            ("3d", (5, 3, 2), False),
        ]
    )
    def test_bitwise_or_scalar(self, _, shape, scalar):
        class bitwise_or(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_or.Scalar(tensor, scalar)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), False),
            ("3d", (5, 3, 2), False),
        ]
    )
    def test_bitwise_or_scalar_tensor(self, _, shape, scalar):
        class bitwise_or(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_or.Scalar_Tensor(scalar, tensor)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_or(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestBitwiseOrTrueScalar(TestCase):
    """`x | True` must stay in PyTorch and still give the right answer.

    DispatchTestCase never consults the capability validator, so these go
    through the full compile pipeline and check that the partitioner built no
    TensorRT engine at all.
    """

    @parameterized.expand(
        [
            ("scalar_2d", (5, 3), False),
            ("scalar_tensor_3d", (5, 3, 2), True),
        ]
    )
    def test_falls_back(self, _, shape, scalar_first):
        class bitwise_or(nn.Module):
            def forward(self, tensor):
                if scalar_first:
                    return torch.ops.aten.bitwise_or.Scalar_Tensor(True, tensor)
                return torch.ops.aten.bitwise_or.Scalar(tensor, True)

        mod = bitwise_or().eval().cuda()
        inputs = [torch.randint(0, 2, shape, dtype=bool).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 0)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


if __name__ == "__main__":
    run_tests()
