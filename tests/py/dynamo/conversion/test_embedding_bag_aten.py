import torch
import torch_tensorrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.utils import ATOL, RTOL

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

    @parameterized.expand(
        [
            param(
                # 1d_indices_mode_0_with_per_sample_weights
                # weights is for compile
                weights=torch.randn((5, 2), dtype=torch.float32),
                # weights_1 is for inference
                weights_1=torch.randn((6, 2), dtype=torch.float32),
                dynamic_shapes={
                    "weights": {0: torch.export.Dim("dyn_dim", min=2, max=6)},
                    "indices": {},
                    "offsets": {},
                },
                indices=torch.tensor([1, 2, 4], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
                mode=0,
                per_sample_weights=torch.randn((3,), dtype=torch.float32),
            ),
            param(
                # 1d_indices_mode_1_without_per_sample_weights
                # weights is for compile
                weights=torch.randn((5, 2), dtype=torch.float32),
                # weights_1 is for inference
                weights_1=torch.randn((6, 3), dtype=torch.float32),
                dynamic_shapes={
                    "weights": {
                        0: torch.export.Dim("dyn_dim", min=2, max=8),
                        1: torch.export.Dim("dyn_dim_1", min=1, max=3),
                    },
                    "indices": {},
                    "offsets": {},
                },
                indices=torch.tensor([1, 2, 4, 2, 3, 4], dtype=torch.int32),
                offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
                mode=1,
                per_sample_weights=None,
            ),
        ]
    )
    def test_embedding_bag_with_weights_dynamic_shape(
        self,
        weights,
        weights_1,
        dynamic_shapes,
        indices,
        offsets,
        mode,
        per_sample_weights,
    ):
        class EmbeddingBag(torch.nn.Module):
            def forward(self, weights, indices, offsets, per_sample_weights=None):
                return torch.ops.aten._embedding_bag.default(
                    weight=weights,
                    indices=indices,
                    offsets=offsets,
                    per_sample_weights=per_sample_weights,
                    scale_grad_by_freq=False,
                    mode=mode,
                    sparse=False,
                    include_last_offset=False,
                    padding_idx=-1,
                )

        if per_sample_weights is None:
            inputs = (weights, indices, offsets)
        else:
            inputs = (weights, indices, offsets, per_sample_weights)
        mod = EmbeddingBag()

        if per_sample_weights is not None:
            dynamic_shapes["per_sample_weights"] = {}
        fx_mod = torch.export.export(mod, inputs, dynamic_shapes=dynamic_shapes)
        trt_mod = torch_tensorrt.dynamo.compile(
            fx_mod,
            inputs=inputs,
            enable_precisions=torch.float32,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        # use the inputs with different shape to inference:
        if per_sample_weights is None:
            inputs = (weights_1, indices, offsets)
        else:
            inputs = (weights_1, indices, offsets, per_sample_weights)

        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())
            ref_outputs = mod(*cuda_inputs)
            outputs = trt_mod(*cuda_inputs)
            for out, ref in zip(outputs, ref_outputs):
                torch.testing.assert_close(
                    out,
                    ref,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=True,
                    check_dtype=True,
                )


if __name__ == "__main__":
    run_tests()
