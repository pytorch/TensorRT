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

        index_input = torch.randint(0, 8, (3, 3, 4, 10), dtype=torch.int32)
        input_specs = [
            Input(
                dtype=torch.int32,
                min_shape=(1, 1, 1, 1),
                opt_shape=(2, 3, 4, 5),
                max_shape=(3, 3, 4, 10),
                torch_tensor=torch.tensor(index_input, dtype=torch.int32).cuda(),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(1, 1),
                opt_shape=(2, 2),
                max_shape=(8, 10),
                torch_tensor=torch.randn((8, 10), dtype=torch.float32).cuda(),
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestEmbedding(),
            input_specs,
            use_example_tensors=False,
        )


if __name__ == "__main__":
    run_tests()
