import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestExpandConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (2, 1), (2, 3)),
            ("3d_dim", (2, 1, 1), (2, 3, 4)),
            ("4d_dim", (2, 1, 1, 1), (2, 3, 4, 5)),
            ("keep_dim", (2, 1, 5, 5), (2, 3, -1, -1)),
            ("different_ranks", (1, 5, 7), (2, 3, -1, -1)),
        ]
    )
    def test_expand(self, _, input_shape, expanded_shape):
        class Expand(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, expanded_shape)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Expand(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_dim", (2, 1), (4, 1), (6, 1), (-1, 3)),
            ("3d_dim", (2, 1, 1), (4, 1, 1), (6, 1, 1), (-1, 3, 4)),
            ("4d_dim", (1, 1, 1, 1), (3, 1, 1, 1), (5, 1, 1, 1), (-1, 2, 3, 6)),
            ("keep_dim", (2, 1, 5, 5), (4, 1, 5, 5), (6, 1, 5, 5), (-1, 3, -1, -1)),
            ("different_ranks", (1, 2, 1), (1, 2, 1), (2, 2, 1), (2, -1, -1, -1)),
        ]
    )
    def test_expand_dynamic_input(
        self, _, min_shape, opt_shape, max_shape, expanded_shape
    ):
        class ExpandInputDynamic(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, expanded_shape)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ExpandInputDynamic(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("3d_dim", (4, 1, 768), (1, 1, 768)),
        ]
    )
    def test_expand_dynamic_target_shape(self, _, input_shape, weight_shape):
        class ExpandTargetDynamic(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.cls_token = torch.nn.Parameter(torch.randn(weight_shape).cuda())

            def forward(self, x):
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                embeddings = torch.cat((cls_tokens, x), dim=0)
                return embeddings

        input_specs = [
            Input(dtype=torch.float32, shape=input_shape),
        ]
        self.run_test_with_dynamic_shape(
            ExpandTargetDynamic(), input_specs, use_dynamo_tracer=True
        )


if __name__ == "__main__":
    run_tests()
