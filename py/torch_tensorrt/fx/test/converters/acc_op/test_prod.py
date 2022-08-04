import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec

# NOTE torch.prod will only accept one dim unlike other reduce ops which accept tuples


class TestProdConverter(AccTestCase):
    @parameterized.expand(
        [
            (
                f"{acc_ops.prod.__name__}_dim0_keepdim",
                0,
                True,
                torch.prod,
                acc_ops.prod,
            ),
            (
                f"{acc_ops.prod.__name__}_dim0_no_keepdim",
                0,
                False,
                torch.prod,
                acc_ops.prod,
            ),
            (
                f"{acc_ops.prod.__name__}_dim1_keepdim",
                1,
                True,
                torch.prod,
                acc_ops.prod,
            ),
            (
                f"{acc_ops.prod.__name__}_dim1_no_keepdim",
                1,
                False,
                torch.prod,
                acc_ops.prod,
            ),
            (
                f"{acc_ops.prod.__name__}_dim1_keepdim",
                2,
                True,
                torch.prod,
                acc_ops.prod,
            ),
            (
                f"{acc_ops.prod.__name__}_dim1_no_keepdim",
                2,
                False,
                torch.prod,
                acc_ops.prod,
            ),
        ]
    )
    def test_prod(self, test_name, dim, keepdim, op, expected_acc_op):
        class Prod(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return op(x, dim=self.dim, keepdim=self.keepdim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Prod(dim, keepdim),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [(f"{acc_ops.prod.__name__}_no_dim_no_keepdim", torch.prod, acc_ops.prod)]
    )
    def test_prod_all_dims(
        self,
        test_name,
        op,
        expected_acc_op,
    ):
        class Prod(torch.nn.Module):
            def forward(self, x):
                return op(x)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Prod(),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=False,
        )

    def test_prod_all_dims_with_dynamic_shape(
        self,
        op=torch.prod,
    ):
        class Prod(torch.nn.Module):
            def forward(self, x):
                return op(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Prod(), input_specs, expected_ops={acc_ops.prod}
        )


if __name__ == "__main__":
    run_tests()
