import torch
from harness import DispatchTestCase
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests


class TestEmbeddingBagConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # 1D input
            param(
                test_name="1d_indices_1",
                weight=torch.randn((10, 3), dtype=torch.float32),
                indices=torch.tensor(
                    [1, 2, 4, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 4, 3, 2],
                    dtype=torch.int32,
                ),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            # param(
            #     test_name="1d_indices_2",
            #     weight=torch.randn((10, 3), dtype=torch.float32),
            #     indices=torch.tensor([1, 2, 4, 5, 4, 3], dtype=torch.int32),
            #     offsets=torch.tensor([0, 5], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=0,
            #     sparse=False,
            #     per_sample_weights=torch.randn((6,)),
            #     include_last_offset=False,
            #     padding_idx=-1,
            # ),
            # param(
            #     test_name="1d_indices_3",
            #     weight=torch.randn((10, 3), dtype=torch.float32),
            #     indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
            #     offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=2,
            #     sparse=False,
            #     per_sample_weights=None,
            #     include_last_offset=False,
            #     padding_idx=-1,
            # ),
            # 2D input
            # param(
            #     test_name="2d_indices_1",
            #     weight=torch.randn((5, 10), dtype=torch.float32),
            #     indices=torch.tensor([[3, 1], [4, 3]], dtype=torch.int32),
            #     offsets=torch.tensor([0, 1], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=0,
            #     sparse=False,
            #     per_sample_weights=torch.randn((4,)),
            #     include_last_offset=False,
            #     padding_idx=-1,
            # ),
            # param(
            #     test_name="2d_indices_3",
            #     weight=torch.tensor([
            #         [0.0, 0.0, 0.0],
            #         [1.0, 1.0, 1.0],
            #         [2.0, 2.0, 2.0],
            #         [3.0, 3.0, 3.0],
            #         [4.0, 4.0, 4.0],
            #         [5.0, 5.0, 5.0],
            #     ], dtype=torch.float32),
            #     indices=torch.tensor([[0, 2, 1], [3, 5, 4]], dtype=torch.int32),
            #     offsets=torch.tensor([0, 1], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=2,
            #     sparse=False,
            #     per_sample_weights=None,
            #     include_last_offset=False,
            #     padding_idx=-1,
            # ),
            # param(
            #     test_name="2d_indices_2",
            #     weight=torch.randn((5, 5), dtype=torch.float32),
            #     indices=torch.tensor([[3, 1, 2], [4, 2, 3]], dtype=torch.int32),
            #     offsets=torch.tensor([0, 2], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=1,
            #     sparse=False,
            #     per_sample_weights=None,
            #     include_last_offset=False,
            #     padding_idx=-1,
            # ),
            # param(
            #     test_name="2d_indices_2",
            #     weight=torch.randn((5, 10), dtype=torch.float32),
            #     indices=torch.tensor([[3, 1, 2, 4], [4, 1, 3, 1]], dtype=torch.int32),
            #     offsets=torch.tensor([0, 2], dtype=torch.int32),
            #     scale_grad_by_freq=False,
            #     mode=0,
            #     sparse=False,
            #     per_sample_weights=torch.randn((8,)),
            #     include_last_offset=True,
            #     padding_idx=-1,
            # ),
        ]
    )
    def test_embedding_bag(
        self,
        test_name,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    ):
        class TestEmbeddingBag(torch.nn.Module):
            def forward(self, weight, indices, offsets):
                return torch.ops.aten._embedding_bag.default(
                    weight,
                    indices,
                    offsets,
                    scale_grad_by_freq,
                    mode,
                    sparse,
                    per_sample_weights,
                    include_last_offset,
                    padding_idx,
                )[0]

        self.run_test(
            TestEmbeddingBag(),
            inputs=[weight, indices, offsets],
            # use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
