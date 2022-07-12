import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestChunkConverter(AccTestCase):
    @parameterized.expand(
        [
            ("chunk", 3, 1),
            ("chunk", 2000, 2),
            ("chunk", 3, -2),
        ]
    )
    def test_chunk(self, _, chunk, dim):
        class Chunk(nn.Module):
            def forward(self, x):
                return x.chunk(chunk, dim)[0]

        inputs = [torch.randn(3, 10, 20)]
        self.run_test(
            Chunk(),
            inputs,
            expected_ops={acc_ops.chunk},
        )

    @parameterized.expand(
        [
            ("chunk", 3, 1),
            ("chunk", 2000, 1),
            ("chunk", 3, -2),
        ]
    )
    def test_chunk_with_dynamic_shape(self, _, chunk, dim):
        class Chunk(nn.Module):
            def forward(self, x):
                return x.chunk(chunk, dim)[0]

        input_specs = [
            InputTensorSpec(
                shape=(-1, 10, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 10, 20), (5, 10, 20), (10, 10, 20))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Chunk(), input_specs, expected_ops={acc_ops.chunk}
        )

    # Testing with (-1, -1, -1, -1) results in Error: AssertionError: Can't chunk on dynamic shape dimension!
    @parameterized.expand(
        [
            ("chunk", 3, 1),
            ("chunk", 2000, 1),
            ("chunk", 3, -2),
        ]
    )
    def test_chunk_with_dynamic_shape_four_dimensions(self, _, chunk, dim):
        class Chunk(nn.Module):
            def forward(self, x):
                return x.chunk(chunk, dim)[0]

        input_specs = [
            InputTensorSpec(
                shape=(-1, 1, 3, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 5), (3, 1, 3, 5), (5, 1, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Chunk(), input_specs, expected_ops={acc_ops.chunk}
        )


if __name__ == "__main__":
    run_tests()
