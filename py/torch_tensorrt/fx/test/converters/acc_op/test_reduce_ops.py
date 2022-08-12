import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec

reduce_ops = [(torch.sum, acc_ops.sum), (torch.mean, acc_ops.mean)]


class TestReduceConverter(AccTestCase):
    @parameterized.expand(
        case
        for op, acc_op in reduce_ops
        for case in [
            (f"{acc_op.__name__}_single_dim_no_keepdim", 1, False, op, acc_op),
            (f"{acc_op.__name__}_single_dim_keepdim", 1, True, op, acc_op),
            (f"{acc_op.__name__}_two_dim_no_keepdim", (1, 2), False, op, acc_op),
            (f"{acc_op.__name__}_two_dim_keepdim", (1, 2), True, op, acc_op),
            (f"{acc_op.__name__}_three_dim_no_keepdim", (1, 2, 3), False, op, acc_op),
            (f"{acc_op.__name__}_three_dim_keepdim", (1, 2, 3), True, op, acc_op),
            (f"{acc_op.__name__}_dim0_keepdim", 0, True, op, acc_op),
            (f"{acc_op.__name__}_dim0_no_keepdim", 0, False, op, acc_op),
            (f"{acc_op.__name__}_neg_single_dim_no_keepdim", -1, False, op, acc_op),
            (f"{acc_op.__name__}_neg_single_dim_keepdim", -1, True, op, acc_op),
            (f"{acc_op.__name__}_neg_two_dim_no_keepdim", (-1, -2), False, op, acc_op),
            (f"{acc_op.__name__}_neg_two_dim_keepdim", (-1, -2), True, op, acc_op),
            (
                f"{acc_op.__name__}_neg_pos_two_dim_no_keepdim",
                (-1, 1),
                False,
                op,
                acc_op,
            ),
            (f"{acc_op.__name__}_neg_pos_two_dim_keepdim", (-1, 1), True, op, acc_op),
        ]
    )
    def test_reduce(self, test_name, dim, keepdim, op, expected_acc_op):
        class Reduce(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return op(x, dim=self.dim, keepdim=self.keepdim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Reduce(dim, keepdim),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            (f"{acc_op.__name__}_no_dim_no_keepdim", op, acc_op)
            for op, acc_op in reduce_ops
        ]
    )
    def test_reduce_all_dims(
        self,
        test_name,
        op,
        expected_acc_op,
    ):
        class Reduce(torch.nn.Module):
            def forward(self, x):
                return op(x)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Reduce(),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            (f"{acc_op.__name__}_no_dim_no_keepdim", op, acc_op)
            for op, acc_op in reduce_ops
        ]
    )
    def test_reduce_all_dims_with_dynamic_shape_four_dimensions(
        self,
        test_name,
        op,
        expected_acc_op,
    ):
        class Reduce(torch.nn.Module):
            def forward(self, x):
                return op(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Reduce(), input_specs, expected_ops={expected_acc_op}
        )


if __name__ == "__main__":
    run_tests()
