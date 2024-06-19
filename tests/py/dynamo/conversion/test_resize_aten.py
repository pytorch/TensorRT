import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestResizeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 5),),
        ]
    )
    def test_resize_1d_input_float(self, target_shape):
        class Resize(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        inputs = [torch.randn(5)]
        self.run_test(
            Resize(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 5),),
        ]
    )
    def test_resize_1d_input_int(self, target_shape):
        class Resize(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        inputs = [torch.randint(1, 5, (5,))]
        self.run_test(
            Resize(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((4, 4),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 5),),
        ]
    )
    def test_resize_2d_input_float(self, target_shape):
        class Resize(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        inputs = [torch.randn(4, 4)]
        self.run_test(
            Resize(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((4, 4),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 5),),
        ]
    )
    def test_resize_2d_input_int(self, target_shape):
        class Resize(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        inputs = [torch.randint(1, 10, (4, 4))]
        self.run_test(
            Resize(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
