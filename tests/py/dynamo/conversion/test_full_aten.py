import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestFullConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((5,), 2),
            ((5, 3), 0.1),
            ((5, 3, 2), True),
        ]
    )
    def test_full_static(self, shape, fill_value):
        class full(nn.Module):
            def forward(self, x):
                return torch.ops.aten.full.default(shape, fill_value)

        inputs = [torch.randn(1, 1)]
        self.run_test(
            full(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (4,), [3], 11),
            ((3, 5), (3, 7), (3, 10), [3, 7], False),
            ((1, 5), (3, 7), (4, 10), [3, 7], True),
            ((1, 5, 3), (3, 7, 3), (4, 10, 4), [3, 7, 3], 0.11),
        ]
    )
    def test_full_dynamic(self, min_shape, opt_shape, max_shape, data, fill_value):
        class full(nn.Module):
            def forward(self, shape):
                return torch.ops.aten.full.default(shape, fill_value)

        inputs = [
            torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.int64,
                torch_tensor=torch.tensor(data, dtype=torch.int64).cuda(),
                is_shape_tensor=True,
            )
        ]
        self.run_test_with_dynamic_shape(
            full(), inputs, use_example_tensors=False, check_dtype=False
        )

    @parameterized.expand(
        [
            ((1, 5, 3), (3, 7, 3), (4, 10, 4), 0.11),
        ]
    )
    def test_full_dynamic_shape_list(self, min_shape, opt_shape, max_shape, fill_value):
        class full(nn.Module):
            def forward(self, x):
                shape = x.shape[0]
                target_shape = (shape, shape + 1)
                return torch.ops.aten.full.default(target_shape, fill_value)

        inputs = [
            torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.int64,
            )
        ]
        self.run_test_with_dynamic_shape(
            full(),
            inputs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
