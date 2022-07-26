import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase


class TestWhere(AccTestCase):
    @parameterized.expand(
        [
            ("same_shape", (1, 3, 2), (1, 3, 2), (1, 3, 2)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 1)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 2)),
        ]
    )
    def test_where(self, _, condition_shape, x_shape, y_shape):
        class Where(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, condition, x, y):
                return torch.where(condition, x, y)

        inputs = [
            (torch.randn(condition_shape) > 0),
            torch.randn(x_shape),
            torch.ones(y_shape),
        ]
        self.run_test(
            Where(),
            inputs,
            expected_ops={acc_ops.where},
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("same_shape", (1, 3, 2), (1, 3, 2), (1, 3, 2)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 1)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 2)),
        ]
    )
    def test_where_attribute_condition(self, _, condition_shape, x_shape, y_shape):
        class Where(nn.Module):
            def __init__(self, condition_shape):
                super().__init__()
                self.condition = torch.randn(condition_shape) > 0

            def forward(self, x, y):
                return torch.where(self.condition, x, y)

        inputs = [torch.randn(x_shape), torch.ones(y_shape)]
        self.run_test(
            Where(condition_shape),
            inputs,
            expected_ops={acc_ops.where},
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("same_shape", (1, 3, 2), (1, 3, 2), (1, 3, 2)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 1)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 2)),
        ]
    )
    def test_where_attribute_condition_x(self, _, condition_shape, x_shape, y_shape):
        class Where(nn.Module):
            def __init__(self, condition_shape, x_shape):
                super().__init__()
                self.condition = torch.randn(condition_shape) > 0
                self.x = torch.randn(x_shape)

            def forward(self, y):
                return torch.where(self.condition, self.x, y)

        inputs = [torch.ones(y_shape)]
        self.run_test(
            Where(condition_shape, x_shape),
            inputs,
            expected_ops={acc_ops.where},
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("same_shape", (1, 3, 2), (1, 3, 2), (1, 3, 2)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 1)),
            ("broadcast_shape", (1, 3, 2), (1, 1, 1), (1, 1, 2)),
        ]
    )
    def test_where_attribute_x_y(self, _, condition_shape, x_shape, y_shape):
        class Where(nn.Module):
            def __init__(self, x_shape, y_shape):
                super().__init__()

                self.x = torch.randn(x_shape)
                self.y = torch.ones(y_shape)

            def forward(self, condition):
                return torch.where(condition, self.x, self.y)

        inputs = [(torch.randn(condition_shape) > 0)]
        self.run_test(
            Where(x_shape, y_shape),
            inputs,
            expected_ops={acc_ops.where},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
