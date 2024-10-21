import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestIsInfConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                torch.tensor(
                    [
                        1.23,
                        -4.56,
                        float("inf"),
                        float("-inf"),
                        -100.0,
                        float("nan"),
                        0.13,
                        -0.13,
                        3.14159265,
                    ]
                ),
            ),
        ]
    )
    def test_isinf_float(self, data):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isinf.default(input)

        inputs = [data]
        self.run_test(
            isinf(),
            inputs,
        )

    def test_isinf_dynamic_shape_float(self):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isinf.default(input)

        inputs = [
            Input(
                min_shape=(1, 2, 3),
                opt_shape=(3, 2, 3),
                max_shape=(5, 3, 3),
                dtype=torch.float32,
                torch_tensor=torch.tensor(
                    ([[[2.7, float("-inf"), 1.1], [4.7, -2.3, float("inf")]]]),
                    dtype=torch.float32,
                ).cuda(),
            )
        ]
        self.run_test_with_dynamic_shape(
            isinf(),
            inputs,
            use_example_tensors=False,
        )

    def test_isinf_dynamic_shape_int(self):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isinf.default(input)

        inputs = [
            Input(
                min_shape=(1, 2),
                opt_shape=(3, 2),
                max_shape=(5, 3),
                dtype=torch.int,
                torch_tensor=torch.tensor(([[-3, 2]]), dtype=torch.int).cuda(),
            )
        ]
        self.run_test_with_dynamic_shape(
            isinf(),
            inputs,
            use_example_tensors=False,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_isinf_int(self, input_shape, dtype, low, high):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.isinf.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            isinf(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
