import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestTruncConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,),),
            ((1, 20),),
            ((2, 3, 4),),
            ((2, 3, 4, 5),),
        ]
    )
    def test_trunc_float(self, shape):
        class Trunc(nn.Module):
            def forward(self, input):
                return torch.ops.aten.trunc.default(input)

        inputs = [torch.randn(shape)]
        self.run_test(
            Trunc(),
            inputs,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((10,),),
            ((1, 20),),
            ((2, 3, 4),),
            ((2, 3, 4, 5),),
        ]
    )
    def test_trunc_int(self, shape):
        class Trunc(nn.Module):
            def forward(self, input):
                return torch.ops.aten.trunc.default(input)

        inputs = [torch.randint(-10, 10, shape, dtype=torch.int32)]
        self.run_test(
            Trunc(),
            inputs,
            enable_passes=True,
        )


class TestTruncConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_int32",
                (1, 1, 1),
                (2, 2, 2),
                (3, 4, 5),
                torch.int32,
                False,
            ),
            (
                "3d_dynamic_float32",
                (2, 1, 1),
                (2, 2, 2),
                (2, 4, 5),
                torch.float32,
                True,
            ),
        ]
    )
    def test_trunc_dynamic(
        self, _, min_shape, opt_shape, max_shape, type, enable_passes
    ):
        class Trunc(nn.Module):
            def forward(self, input):
                return torch.ops.aten.trunc.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Trunc(),
            input_specs,
            enable_passes=enable_passes,
        )


if __name__ == "__main__":
    run_tests()
