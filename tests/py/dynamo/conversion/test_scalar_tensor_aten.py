import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestScalarTensorConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (-2.00001,),
            (-1.3,),
            (-0.0,),
            (1.0,),
            (2.99,),
        ]
    )
    def test_scalar_tensor_float(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    @parameterized.expand(
        [
            (-9999,),
            (-1,),
            (0,),
            (2,),
            (99999,),
        ]
    )
    def test_scalar_tensor_int(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_scalar_tensor_bool(self, scalar):
        class ScalarTensor(nn.Module):
            def forward(self):
                return torch.ops.aten.scalar_tensor.default(scalar)

        inputs = []
        self.run_test(
            ScalarTensor(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
