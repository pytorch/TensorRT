import unittest

import torch

import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


@unittest.skip(
    "Current implementation is limited. All implementations in hf use int64. T113156424"
)
class TestEmbeddingConverter(AccTestCase):
    @parameterized.expand(
        [
            param(
                test_name="1d_indices",
                indices_tensor=torch.tensor([3, 1, 2]),
                weights_tensor=torch.randn(5, 10),
            ),
            param(
                test_name="2d_indices",
                indices_tensor=torch.tensor([[3, 1, 2], [4, 1, 3]]),
                weights_tensor=torch.randn(5, 10),
            ),
            param(
                test_name="3d_indices",
                indices_tensor=torch.tensor([[[0, 1], [2, 3]], [[3, 4], [4, 0]]]),
                weights_tensor=torch.randn(5, 10),
            ),
        ]
    )
    def test_embedding(
        self,
        test_name,
        indices_tensor,
        weights_tensor,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        class TestEmbedding(torch.nn.Module):
            def forward(self, indices, weights):
                return torch.nn.functional.embedding(
                    input=indices,
                    weight=weights,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    norm_type=norm_type,
                    scale_grad_by_freq=scale_grad_by_freq,
                    sparse=sparse,
                )

        self.run_test(
            TestEmbedding(),
            inputs=[indices_tensor.int(), weights_tensor.float()],
            expected_ops={acc_ops.embedding},
            test_implicit_batch_dim=False,
            test_explicit_batch_dim=True,
        )

    def test_embedding_with_dynamic_shape_four_dimensions(
        self,
        test_name,
        indices_tensor,
        weights_tensor,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        class TestEmbedding(torch.nn.Module):
            def forward(self, indices, weights):
                return torch.nn.functional.embedding(
                    input=indices,
                    weight=weights,
                    padding_idx=padding_idx,
                    max_norm=max_norm,
                    norm_type=norm_type,
                    scale_grad_by_freq=scale_grad_by_freq,
                    sparse=sparse,
                )

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestEmbedding(), input_specs, expected_ops={acc_ops.embedding}
        )


if __name__ == "__main__":
    run_tests()
