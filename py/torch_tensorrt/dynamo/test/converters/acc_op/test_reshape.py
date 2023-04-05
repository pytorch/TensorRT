import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestReshapeConverter(AccTestCase):
    @parameterized.expand(
        [
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    def test_reshape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(TestModule(target_shape), inputs, expected_ops={acc_ops.reshape})

    @parameterized.expand(
        [
            ((-1, 2),),
            ((1, 2, -1),),
        ]
    )
    def test_reshape_with_dynamic_shape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(target_shape), input_specs, expected_ops={acc_ops.reshape}
        )

    @parameterized.expand(
        [
            ((-1, 2),),
            ((1, 2, -1),),
        ]
    )
    def test_reshape_with_dynamic_shape_with_four_dimensions(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(target_shape), input_specs, expected_ops={acc_ops.reshape}
        )

    def test_reshape_with_dynamic_shape_size(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                shape_y = y.shape
                t = shape_y[1]
                return torch.reshape(x, [-1, t, 3])

        input_specs = [
            InputTensorSpec(
                shape=(-1, 5, 6),
                dtype=torch.float32,
                shape_ranges=[((1, 5, 6), (2, 5, 6), (3, 5, 6))],
            ),
            InputTensorSpec(
                shape=(-1, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 5), (1, 5), (3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.reshape}
        )

    def test_reshape_with_dynamic_shape_mul(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y, z):
                t = 8000
                a = torch.reshape(x, [-1, t, 64])
                b = torch.reshape(y, [-1, t, 64])
                c = torch.reshape(z, [-1, t, 64])
                d = a + b + c
                return d

        input_specs = [
            InputTensorSpec(
                shape=(-1, 42, 512),
                dtype=torch.float32,
                shape_ranges=[((1, 42, 512), (1000, 42, 512), (1000, 42, 512))],
            ),
            InputTensorSpec(
                shape=(-1, 42, 512),
                dtype=torch.float32,
                shape_ranges=[((1, 42, 512), (1000, 42, 512), (1000, 42, 512))],
            ),
            InputTensorSpec(
                shape=(-1, 42, 512),
                dtype=torch.float32,
                shape_ranges=[((1, 42, 512), (1000, 42, 512), (1000, 42, 512))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.reshape}
        )


if __name__ == "__main__":
    run_tests()
