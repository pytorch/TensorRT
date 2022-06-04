import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


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


if __name__ == "__main__":
    run_tests()
