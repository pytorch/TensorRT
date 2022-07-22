import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestMinConverter(AccTestCase):
    @parameterized.expand(
        [
            ("dim0_keepdim", 0, True, torch.randn(2, 2, 3)),
            ("dim1_keepdim", 1, True, torch.randn(2, 2, 3)),
            ("dim2_keepdim", 2, True, torch.randn(2, 2, 3)),
            ("dim3_keepdim", 3, True, torch.randn(2, 2, 3, 3)),
            ("dim2_no_keepdim", 2, False, torch.randn(2, 2, 3)),
            ("dim1_no_keepdim", 1, False, torch.randn(2, 2, 3)),
            ("dim0_no_keepdim", 0, False, torch.randn(2, 2, 3)),
        ]
    )
    def test_min_dim_reduce(self, test_name, dim, keepdim, input):
        class MinDimReduce(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.min(x, self.dim, self.keepdim)

        inputs = [input]
        self.run_test(
            MinDimReduce(dim, keepdim),
            inputs,
            expected_ops={acc_ops.min_dim_reduce},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            ("no_dim_no_keepdim"),
        ]
    )
    def test_min_full_reduce(
        self,
        test_name,
    ):
        class MinFullReduce(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.min(x)

        inputs = [torch.randn(3, 2, 3, 3)]
        self.run_test(
            MinFullReduce(),
            inputs,
            expected_ops={acc_ops.min_full_reduce},
            # We can't do a full reduce over the batch dimension
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("min_method_no_dim_no_keepdim"),
            ("min_method_no_dim_no_keepdim"),
        ]
    )
    def test_min_method(self, test_name):
        class MinMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, other):
                return input.min(other)

        inputs = [torch.randn(3, 4), torch.randn(3, 4)]
        self.run_test(MinMethod(), inputs, expected_ops={acc_ops.minimum})


class TestMinConverterWithDynamicShape(AccTestCase):
    @parameterized.expand(
        [
            ("dim0_keepdim", 0, True),
            ("dim1_keepdim", 1, True),
            ("dim2_keepdim", 2, True),
            ("dim3_keepdim", 3, True),
        ]
    )
    def test_min_dim_reduce(self, test_name, dim, keepdim):
        class MinDimReduce(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.min(x, dim, keepdim)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            MinDimReduce(), input_specs, expected_ops={acc_ops.min_dim_reduce}
        )

    def test_min_full_reduce(
        self,
    ):
        class MinFullReduce(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.min(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            MinFullReduce(), input_specs, expected_ops={acc_ops.min_full_reduce}
        )

    def test_min_method(self):
        class MinMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, other):
                return input.min(other)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            MinMethod(), input_specs, expected_ops={acc_ops.minimum}
        )


if __name__ == "__main__":
    run_tests()
