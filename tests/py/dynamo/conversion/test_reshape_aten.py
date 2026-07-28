import unittest

import tensorrt as trt
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt._utils import is_tensorrt_version_supported

from .harness import DispatchTestCase


class TestReshapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((-1,),),
            ((20,),),
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    @unittest.skipIf(
        not is_tensorrt_version_supported("8.5"),
        "Shape tensor supported well in TensorRT 8.5 and later",
    )
    def test_reshape(self, target_shape):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.view.default(x, target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(
            Reshape(),
            inputs,
        )

    @parameterized.expand(
        [
            ((-1,),),
            ((-1, 10),),
            ((-1, 5),),
            ((2, 2, -1),),
        ]
    )
    @unittest.skipIf(
        not is_tensorrt_version_supported("8.5"),
        "Shape tensor supported well in TensorRT 8.5 and later",
    )
    def test_reshape_with_dynamic_shape(self, target_shape):
        class Reshape(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.view.default(x, target_shape)

        input_specs = [
            Input(
                shape=(-1, 2, 5),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 5), (10, 2, 5), (10, 2, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Reshape(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
