import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.export import Dim
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.utils import ATOL, RTOL

from .harness import DispatchTestCase


class TestBitwiseAndConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 3), (2, 3)),
            ("3d", (5, 3, 2), (5, 3, 2)),
            ("3d_broadcast", (2, 3), (2, 1, 3)),
            ("4d_broadcast_1", (2, 3), (1, 2, 1, 3)),
            ("4d_broadcast_2", (2, 3), (2, 2, 2, 3)),
        ]
    )
    def test_bitwise_and_tensor(self, _, lhs_shape, rhs_shape):
        class bitwise_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_and.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 2, lhs_shape, dtype=bool),
            torch.randint(0, 2, rhs_shape, dtype=bool),
        ]
        self.run_test(
            bitwise_and(),
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
    def test_bitwise_and_tensor_dynamic_shape(
        self,
        _,
        lhs_min_shape,
        lhs_opt_shape,
        lhs_max_shape,
        rhs_min_shape,
        rhs_opt_shape,
        rhs_max_shape,
    ):
        class bitwise_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_and.Tensor(lhs_val, rhs_val)

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
            bitwise_and(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
            use_example_tensors=False,
        )

    # Only True is here; `x & False` is handled by TestBitwiseAndFalseScalar
    # below, because the validator keeps it out of TensorRT.
    @parameterized.expand(
        [
            ("2d", (5, 3), True),
            ("3d", (5, 3, 2), True),
        ]
    )
    def test_bitwise_and_scalar(self, _, shape, scalar):
        class bitwise_and(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_and.Scalar(tensor, scalar)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_and(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), True),
            ("3d", (5, 3, 2), True),
        ]
    )
    def test_bitwise_and_scalar_tensor(self, _, shape, scalar):
        class bitwise_and(nn.Module):
            def forward(self, tensor):
                return torch.ops.aten.bitwise_and.Scalar_Tensor(scalar, tensor)

        inputs = [
            torch.randint(0, 2, shape, dtype=bool),
        ]
        self.run_test(
            bitwise_and(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )

    # this test case is to test the bitwise_and with different ranks
    # it cannot use the normal test_with_dynamic_shape due to the
    # torch_tensorrt.dynamo.trace doesn't automatically handle it
    # hence has to manually export the graph and run the test.
    def test_bitwise_and_dynamic_shape_with_different_ranks(self):
        class bitwise_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.bitwise_and.Tensor(lhs_val, rhs_val)

        dyn_dim = Dim("dyn_dim", min=2, max=6)
        inputs = (
            torch.randint(0, 2, (2, 4, 2), dtype=bool),
            torch.randint(0, 2, (4, 2), dtype=bool),
        )
        mod = bitwise_and()
        fx_mod = torch.export.export(
            mod, inputs, dynamic_shapes=({1: dyn_dim}, {0: dyn_dim})
        )
        trt_mod = torch_tensorrt.dynamo.compile(
            fx_mod,
            inputs=inputs,
            enable_precisions={torch.bool},
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())
            ref_outputs = mod(*cuda_inputs)
            outputs = trt_mod(*cuda_inputs)
            for out, ref in zip(outputs, ref_outputs):
                torch.testing.assert_close(
                    out,
                    ref,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=True,
                    check_dtype=True,
                )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestBitwiseAndFalseScalar(TestCase):
    """`x & False` must stay in PyTorch and still give the right answer.

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
        class bitwise_and(nn.Module):
            def forward(self, tensor):
                if scalar_first:
                    return torch.ops.aten.bitwise_and.Scalar_Tensor(False, tensor)
                return torch.ops.aten.bitwise_and.Scalar(tensor, False)

        mod = bitwise_and().eval().cuda()
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
