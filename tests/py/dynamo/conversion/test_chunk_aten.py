import unittest

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestChunkConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), 3, 0),
            ((3,), 3, 0),
            ((4,), 3, 0),
            ((6,), 3, 0),
            ((3,), 1, -1),
            ((3,), 3, -1),
            ((3,), 4, -1),
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
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 4), 1, 0),
            ((3, 4), 3, 0),
            ((3, 4), 4, 0),
            ((3, 4), 2, -2),
            ((3, 4), 6, -2),
            ((3, 4), 3, 1),
            ((3, 4), 4, 1),
            ((3, 4), 5, -1),
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
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 4, 2), 1, 0),
            ((3, 4, 2), 3, -3),
            ((3, 4, 2), 3, 1),
            ((3, 4, 2), 4, 1),
            ((3, 4, 2), 6, -2),
            ((3, 4, 2), 1, 2),
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
            use_dynamo_tracer=True,
        )


#######################Dynamic cases#######################
# The tests are skipped for now. Will be addressed once https://github.com/pytorch/pytorch/issues/134663 is addressed
@unittest.skip(
    "Pending aten.split dynamic input torch.export guard bug. Issue- https://github.com/pytorch/pytorch/issues/134663"
)
class TestChunkDynamicConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), (1,), (3,), 3, 0),
            ((3,), (3,), (4,), 3, 0),
            ((4,), (4,), (6,), 3, 0),
            ((6,), (6,), (9,), 3, 0),
            ((3,), (3,), (4,), 1, -1),
            ((3,), (3,), (4,), 3, -1),
            ((3,), (3,), (4,), 4, -1),
        ]
    )
    def test_chunk_1D(self, min_shape, opt_shape, max_shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestChunk(),
            input_specs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 4), (3, 4), (4, 4), 1, 0),
            ((3, 4), (3, 4), (4, 4), 3, 0),
            ((3, 4), (3, 4), (4, 4), 4, 0),
            ((3, 4), (3, 4), (4, 4), 2, -2),
            ((3, 4), (3, 4), (4, 4), 6, -2),
            ((3, 4), (3, 4), (4, 4), 3, 1),
            ((3, 4), (3, 4), (4, 4), 4, 1),
            ((3, 4), (3, 4), (4, 4), 5, -1),
        ]
    )
    def test_chunk_2D(self, min_shape, opt_shape, max_shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestChunk(),
            input_specs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 1, 0),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 3, -3),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 3, 1),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 4, 1),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 6, -2),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 1, 2),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 3, -1),
            ((3, 4, 2), (3, 4, 2), (4, 4, 2), 4, -1),
        ]
    )
    def test_chunk_3D(self, min_shape, opt_shape, max_shape, chunks, dim):
        class TestChunk(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.chunk.default(input, chunks, dim)
                return out

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestChunk(),
            input_specs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
