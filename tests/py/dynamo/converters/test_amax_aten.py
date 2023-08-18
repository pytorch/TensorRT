import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.test_utils import DispatchTestCase


class TestAmaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
        ]
    )
    def test_amax_dim_int_int(self, input_shape, dim, keep_dims, dtype):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.amax(x, dim=dim, keepdim=keep_dims)

        inputs = [torch.randn(*input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
            inputs,
            expected_ops={torch.ops.aten.amax.default},
        )

    @parameterized.expand(
        [
            ((3, 2, 4), [1], True),
            ((2, 1, 4, 5), [0, 3], True),
            ((2, 3, 4, 5), [0, 1, 2, 3], False),
            ((6, 7, 5, 4, 5), [1, 3, 4], False),
        ]
    )
    def test_amax_dim_tuple_int(self, input_shape, dim, keep_dims, dtype):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.amax(x, dim=dim, keepdim=keep_dims)

        inputs = [torch.randn(*input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
            inputs,
            expected_ops={torch.ops.aten.amax.default},
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 3, True, torch.int, -10, 10),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_amax_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.amax(x, dim=dim, keepdim=keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
            inputs,
            expected_ops={torch.ops.aten.amax.default},
            check_dtype=False,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), [1], True, torch.int, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int, -10, 10),
            ((2, 3, 4, 5), [0, 1, 2, 3], False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
        ]
    )
    def test_amax_dim_tuple_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.amax(x, dim=dim, keepdim=keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
            inputs,
            expected_ops={torch.ops.aten.amax.default},
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
