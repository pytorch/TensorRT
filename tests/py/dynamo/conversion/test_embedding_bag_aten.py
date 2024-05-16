import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestEmbeddingBagConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # mode=0: sum, mode=1: mean, mode=2: max
            # 1D input
            param(
                test_name="1d_indices_1",
                weight=torch.randn((10, 2), dtype=torch.float16),
                indices=torch.tensor(
                    [1, 2, 4, 5, 4, 3, 2, 6, 8, 1, 2], dtype=torch.int32
                ),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=True,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_2",
                weight=torch.randn((10, 2), dtype=torch.float16),
                indices=torch.tensor(
                    [1, 2, 4, 5, 4, 3, 2, 6, 8, 1, 2], dtype=torch.int32
                ),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=True,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_3",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=2,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_4",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=False,
                per_sample_weights=torch.randn((8,), dtype=torch.float16),
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_5",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 5, 5], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_6",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 5, 5], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=2,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_7",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 8, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_8",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 8, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
        ]
    )
    def test_embedding_bag_with_traversable_offsets(
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
            def forward(self, weight, indices):
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
            inputs=[weight, indices],
            precision=weight.dtype,
            enable_passes=True,
            propagate_shapes=True,
        )

    @parameterized.expand(
        [
            # mode=0: sum, mode=1: mean, mode=2: max
            # 1D input
            param(
                test_name="1d_indices_1",
                weight=torch.randn((10, 2), dtype=torch.float32),
                indices=torch.tensor(
                    [1, 2, 4, 5, 4, 3, 2, 6, 8, 1, 2], dtype=torch.int32
                ),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=True,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_2",
                weight=torch.randn((10, 2), dtype=torch.float32),
                indices=torch.tensor(
                    [1, 2, 4, 5, 4, 3, 2, 6, 8, 1, 2], dtype=torch.int32
                ),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=True,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_3",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=2,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_4",
                weight=torch.randn((10, 4), dtype=torch.float32),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=False,
                per_sample_weights=torch.randn((8,), dtype=torch.float32),
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_5",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 5, 5], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_6",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 5, 5], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=2,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_7",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 8, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=True,
                padding_idx=-1,
            ),
            param(
                test_name="1d_indices_8",
                weight=torch.randn((10, 4), dtype=torch.float16),
                indices=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.int32),
                offsets=torch.tensor([0, 8, 8], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=1,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
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
    def test_embedding_bag_with_ITensor_offsets(
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
            precision=weight.dtype,
            enable_passes=True,
            propagate_shapes=True,
        )

    @parameterized.expand(
        [
            param(
                test_name="dynamic_offsets_1",
                weight=torch.range(0, 29, dtype=torch.float32).reshape(15, 2),
                indices=torch.tensor([i for i in range(15)], dtype=torch.int32),
                offsets=torch.tensor([0, 2], dtype=torch.int32),
                scale_grad_by_freq=False,
                mode=0,
                sparse=False,
                per_sample_weights=None,
                include_last_offset=False,
                padding_idx=-1,
            ),
        ]
    )
    def test_embedding_bag_with_dynamic_offsets(
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
                offsets_list = []
                end = torch.randint(8, 14, (1,))[0]
                for i in range(3, 0, -1):
                    rand_tensor = torch.arange(5, end, step=i, dtype=torch.int32)
                    offsets_list.append(
                        torch.ops.aten.cat.default((offsets, rand_tensor))
                    )

                res = []
                for one_offsets in offsets_list:
                    output = torch.ops.aten._embedding_bag.default(
                        weight,
                        indices,
                        one_offsets,
                        scale_grad_by_freq,
                        mode,
                        sparse,
                        per_sample_weights,
                        include_last_offset,
                        padding_idx,
                    )[0]
                    res.append(output)

                return res

        self.run_test(
            TestEmbeddingBag(),
            inputs=[weight, indices, offsets],
            precision=weight.dtype,
            enable_passes=True,
            propagate_shapes=True,
        )


if __name__ == "__main__":
    run_tests()
