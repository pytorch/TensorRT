import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestChunkConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), 3, 0),
            ((2,), 3, 0),
            ((3,), 3, 0),
            ((4,), 3, 0),
            ((5,), 3, 0),
            ((6,), 3, 0),
            ((3,), 1, -1),
            ((3,), 2, -1),
            ((3,), 3, -1),
            ((3,), 4, -1),
            ((3,), 5, -1),
            ((3,), 6, -1),
        ]
    )
    def test_chunk_1D(self, shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input = [torch.randn(shape)]
        self.run_test(
            TestChunk(),
            input,
        )

    @parameterized.expand(
        [
            ((3, 4), 1, 0),
            ((3, 4), 2, 0),
            ((3, 4), 3, 0),
            ((3, 4), 4, 0),
            ((3, 4), 5, -2),
            ((3, 4), 6, -2),
            ((3, 4), 7, -2),
            ((3, 4), 3, 1),
            ((3, 4), 4, 1),
            ((3, 4), 5, -1),
            ((3, 4), 6, -1),
        ]
    )
    def test_chunk_2D(self, shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input = [torch.randn(shape)]
        self.run_test(
            TestChunk(),
            input,
        )

    @parameterized.expand(
        [
            ((3, 4, 2), 1, 0),
            ((3, 4, 2), 2, 0),
            ((3, 4, 2), 3, -3),
            ((3, 4, 2), 4, -3),
            ((3, 4, 2), 3, 1),
            ((3, 4, 2), 4, 1),
            ((3, 4, 2), 5, -2),
            ((3, 4, 2), 6, -2),
            ((3, 4, 2), 1, 2),
            ((3, 4, 2), 2, 2),
            ((3, 4, 2), 3, -1),
            ((3, 4, 2), 4, -1),
        ]
    )
    def test_chunk_3D(self, shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input = [torch.randn(shape)]
        self.run_test(
            TestChunk(),
            input,
        )


if __name__ == "__main__":
    run_tests()
