import unittest

import tensorrt as trt
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.fx_ts_compat.tools.common_fx2trt import (
    DispatchTestCase,
    InputTensorSpec,
)


class TestReshapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    @unittest.skipIf(
        trt.__version__ < "8.5",
        "Shape tensor supported well in TensorRT 8.5 and later",
    )
    def test_reshape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(
            TestModule(target_shape),
            inputs,
            expected_ops={torch.ops.aten.view.default},
        )

    @parameterized.expand(
        [
            ((-1, 10),),
            ((-1, 5),),
            ((2, 2, -1),),
        ]
    )
    @unittest.skipIf(
        trt.__version__ < "8.5",
        "Shape tensor supported well in TensorRT 8.5 and later",
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
                shape=(-1, 2, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 5), (10, 2, 5), (10, 2, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(target_shape),
            input_specs,
            expected_ops={torch.ops.aten.view.default},
        )

    @unittest.skipIf(
        trt.__version__ < "8.5",
        "Shape tensor supported well in TensorRT 8.5 and later",
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
                shape_ranges=[((1, 5, 6), (3, 5, 6), (3, 5, 6))],
            ),
            InputTensorSpec(
                shape=(-1, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 5), (3, 5), (3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={torch.ops.aten.view.default},
        )


if __name__ == "__main__":
    run_tests()
