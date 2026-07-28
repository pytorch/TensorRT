import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestCdistConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("p_0", (4, 3, 4), 0, 0),
            ("p>0_p<1_1", (10, 3, 5, 2, 6), 0.5, 1),
            ("p>0_p<1_2", (10, 2, 15, 2, 7, 2), 0.5, 1),
            ("p_1", (15, 10, 5), 1, None),
            ("p>1_p<2", (19, 11, 5), 1.5, None),
            ("small_p_2_mode_1", (6, 6, 5), 2.0, 1),
            ("large_p_2_mode_0", (35, 35, 5), 2.0, 0),
            ("p>2", (15, 10, 5), 2.99, None),
            ("p_inf", (5, 15, 5), float("inf"), 0),
        ]
    )
    def test_cdist_float_same_shape(self, name, shape, p, compute_mode):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                return torch.ops.aten._cdist_forward.default(x1, x2, p, compute_mode)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Cdist(),
            inputs,
        )

    @parameterized.expand(
        [
            ("p_0", (1, 5), (2, 3, 5), 0, 0),
            ("p_1", (4, 5), (2, 3, 5), 1, None),
            ("diff_shape_p_0", (2, 5, 4, 5), (2, 5, 8, 5), 0, 2),
            ("diff_shape_p_1", (2, 4, 5), (2, 3, 5), 1, 1),
            ("p>0_p<1", (2, 2, 4, 5), (2, 3, 5), 0.5, None),
            ("p>1_p<2", (5, 2, 12, 5), (2, 3, 5), 1.5, 1),
            ("p_2", (2, 2, 14, 5), (2, 3, 5), 2, 0),
            ("p>2", (2, 2, 4, 5), (2, 10, 5), 2.99, 2),
            ("p_inf", (2, 2, 3, 5), (2, 8, 5), float("inf"), None),
        ]
    )
    def test_cdist_float_broadcast_and_diff_shape(
        self, name, shape_1, shape_2, p, compute_mode
    ):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                return torch.ops.aten._cdist_forward.default(x1, x2, p, compute_mode)

        inputs = [torch.randn(shape_1), torch.randn(shape_2)]
        self.run_test(
            Cdist(),
            inputs,
        )

    @parameterized.expand(
        [
            ("compute_mode_0", (15, 10, 5), (15, 35, 5), 2.0, 0),
            ("compute_mode_1", (35, 35, 5), (35, 45, 5), 2.0, 0),
            ("compute_mode_2", (15, 10, 5), (15, 35, 5), 2.0, 1),
            ("compute_mode_3", (35, 35, 5), (35, 45, 5), 2.0, 2),
            ("p_2_mm_shape_1", (2, 2, 14, 5), (3, 5), 2, 1),
            ("p_2_mm_shape_2", (2, 2, 14, 5), (2, 3, 5), 2, 1),
            ("p_2_mm_shape_3", (2, 2, 14, 5), (2, 2, 3, 5), 2, 1),
        ]
    )
    def test_cdist_p_2_compute_mode(self, name, shape_1, shape_2, p, compute_mode):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                return torch.ops.aten._cdist_forward.default(x1, x2, p, compute_mode)

        inputs = [torch.randn(shape_1), torch.randn(shape_2)]
        self.run_test(Cdist(), inputs)

    @parameterized.expand(
        [
            ("p_2_matmul", (50, 40, 30, 30), (50, 40, 35, 30), 2, 1),
            ("p_2_elementwise_pow", (50, 40, 30, 50), (50, 40, 35, 50), 2, 2),
        ]
    )
    def test_cdist_efficiency_p_2_compute_mode(
        self, name, shape_1, shape_2, p, compute_mode
    ):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                return torch.ops.aten._cdist_forward.default(x1, x2, p, compute_mode)

        inputs = [torch.randn(shape_1), torch.randn(shape_2)]
        self.run_test(Cdist(), inputs)


if __name__ == "__main__":
    run_tests()
