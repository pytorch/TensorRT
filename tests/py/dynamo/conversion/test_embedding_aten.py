import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestEmbeddingConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param(
                test_name="1d_indices",
                indices_tensor=torch.tensor([3, 1, 2], dtype=torch.int32),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=False,
            ),
            param(
                test_name="2d_indices",
                indices_tensor=torch.tensor([[3, 1, 2], [4, 1, 3]], dtype=torch.int32),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=True,
            ),
            param(
                test_name="3d_indices",
                indices_tensor=torch.tensor(
                    [[[0, 1], [2, 3]], [[3, 4], [4, 0]]], dtype=torch.int32
                ),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=True,
            ),
            # int64 indices - TensorRT now supports int64 for gather operations
            param(
                test_name="1d_indices_int64",
                indices_tensor=torch.tensor([3, 1, 2], dtype=torch.int64),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=False,
            ),
            param(
                test_name="2d_indices_int64",
                indices_tensor=torch.tensor([[3, 1, 2], [4, 1, 3]], dtype=torch.int64),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=True,
            ),
            param(
                test_name="3d_indices_int64",
                indices_tensor=torch.tensor(
                    [[[0, 1], [2, 3]], [[3, 4], [4, 0]]], dtype=torch.int64
                ),
                weights_tensor=torch.randn((5, 10), dtype=torch.float32),
                sparse=True,
            ),
        ]
    )
    def test_embedding(
        self,
        test_name,
        indices_tensor,
        weights_tensor,
        padding_idx=-1,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=None,
        sparse=False,
    ):
        class TestEmbedding(torch.nn.Module):
            def forward(self, indices, weights):
                return torch.ops.aten.embedding.default(
                    weights,
                    indices,
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )

        self.run_test(
            TestEmbedding(),
            inputs=[indices_tensor, weights_tensor],
        )

    def test_embedding_with_dynamic_shape_four_dimensions(
        self,
        padding_idx=-1,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=None,
        sparse=None,
    ):
        class TestEmbedding(torch.nn.Module):
            def forward(self, input, weights):
                return torch.ops.aten.embedding.default(
                    weights,
                    input,
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )

        input_specs = [
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.int32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            Input(
                shape=(-1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1), (2, 3), (2, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestEmbedding(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
