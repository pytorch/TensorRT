import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestPdistConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 3), 0),
            ((2, 3), 0.4),
            ((2, 3), 1),
            ((2, 3), 1.5),
            ((3, 4), 2),
            ((3, 4), 2.99),
            ((4, 5), 3),
            ((4, 5), 3.3),
            ((5, 6), float("inf")),
        ]
    )
    def test_pdist_float(self, shape, p):
        class Pdist(nn.Module):
            def forward(self, input):
                return torch.ops.aten._pdist_forward.default(input, p)

        inputs = [torch.randn(shape)]
        self.run_test(
            Pdist(),
            inputs,
        )


class TestDynamicShapePdistConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "dim0_dynamic_dim1_static_p_0",
                (1, 4),
                (2, 4),
                (4, 4),
                0,
            ),
            (
                "dim0_static_dim1_dynamic_p_1",
                (3, 1),
                (3, 2),
                (3, 4),
                1,
            ),
            (
                "dim0_dynamic_dim1_static_p_other",
                (1, 5),
                (2, 5),
                (6, 5),
                0.4,
            ),
            (
                "dim0_dynamic_dim1_dynamic_p_inf",
                (1, 1),
                (2, 2),
                (5, 4),
                float("inf"),
            ),
            # disable this testcase due to https://github.com/pytorch/TensorRT/issues/3898
            # TODO: enable back once the issue is fixed in both rtx 1.2 and tensorrt 10.14
            # (
            #     "dim0_dynamic_dim1_dynamic_p_other",
            #     (2, 1),
            #     (3, 2),
            #     (4, 7),
            #     1.7,
            # ),
        ]
    )
    def test_pdist_float(self, _, min_shape, opt_shape, max_shape, p):
        class Pdist(nn.Module):
            def forward(self, input):
                return torch.ops.aten._pdist_forward.default(input, p)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]

        self.run_test_with_dynamic_shape(Pdist(), input_specs)


if __name__ == "__main__":
    run_tests()
