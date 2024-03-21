import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAdaptiveAvgPoolConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (2, 3),
                2,
            ),
            (
                (2, 8),
                8,
            ),
            (
                (1, 2, 3),
                2,
            ),
            (
                (2, 2, 8),
                16,
            ),
            (
                (2, 3),
                (1,),
            ),
            (
                (2, 3),
                (2,),
            ),
            (
                (2, 8),
                (4,),
            ),
            (
                (2, 8),
                (16,),
            ),
            (
                (2, 3, 1),
                (1,),
            ),
            (
                (2, 3, 2),
                (2,),
            ),
            (
                (2, 3, 4),
                (4,),
            ),
            (
                (2, 2, 32),
                (31,),
            ),
            (
                (2, 2, 32),
                (64,),
            ),
        ]
    )
    def test_adaptive_avg_pool1d(
        self,
        input_shape,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool1d.default(x, output_size)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
            # use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
