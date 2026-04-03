import pytest
import torch
import torch_tensorrt as torchtrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import ENABLED_FEATURES

from .harness import DispatchTestCase

# NOTE: accumulate=True with *duplicate* indices is NOT supported in TRT.
# TensorRT's ScatterMode.ND overwrites on collision — there is no scatter_add
# reduction mode. The current implementation (scatter into zeros + elementwise
# add) only gives correct results when every scattered index is unique.


class TestIndexPutConverter(DispatchTestCase):
    @parameterized.expand(
        [
            param(
                test_name="1d_indices_single",
                source_tensor=torch.zeros([5], dtype=torch.int32),
                indices_tensor=(torch.tensor([0], dtype=torch.int32),),
                value_tensor=torch.tensor([1], dtype=torch.int32),
            ),
            param(
                test_name="1d_indices_multiple",
                source_tensor=torch.zeros([5], dtype=torch.int32),
                indices_tensor=(torch.tensor([0, 3], dtype=torch.int32),),
                value_tensor=torch.tensor([1, 3], dtype=torch.int32),
            ),
            param(
                test_name="2d_indices_single",
                source_tensor=torch.zeros([5, 5], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([3], dtype=torch.int32),
            ),
            param(
                test_name="2d_indices_multiple",
                source_tensor=torch.zeros([5, 5], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0, 2, 2], dtype=torch.int32),
                    torch.tensor([2, 0, 2], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1, 3, 4], dtype=torch.int32),
            ),
            param(
                test_name="3d_indices_single",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([7], dtype=torch.int32),
            ),
            param(
                test_name="3d_indices_multiple",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0, 1, 1], dtype=torch.int32),
                    torch.tensor([1, 2, 1], dtype=torch.int32),
                    torch.tensor([2, 0, 2], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([5, 7, 2], dtype=torch.int32),
            ),
            param(
                test_name="4d_indices_single",
                source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([0], dtype=torch.int32),
                    torch.tensor([1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([5], dtype=torch.int32),
            ),
            param(
                test_name="4d_indices_multiple",
                source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([1, 1], dtype=torch.int32),
                    torch.tensor([1, 0], dtype=torch.int32),
                    torch.tensor([1, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([5, 7], dtype=torch.int32),
            ),
            param(
                test_name="negative_indices",
                source_tensor=torch.zeros([5, 5], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([-1, -2], dtype=torch.int32),
                    torch.tensor([2, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1, 3], dtype=torch.int32),
            ),
            param(
                test_name="mixed_indices",
                source_tensor=torch.zeros([4, 4], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0, 1, -1, -2], dtype=torch.int32),
                    torch.tensor([0, -1, 2, 1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([2, 4, 6, 8], dtype=torch.int32),
            ),
            param(
                test_name="1d_indices_float",
                source_tensor=torch.zeros([5], dtype=torch.float32),
                indices_tensor=(torch.tensor([0, 3], dtype=torch.int32),),
                value_tensor=torch.tensor([1.5, 3.5], dtype=torch.float32),
            ),
            param(
                test_name="2d_indices_float",
                source_tensor=torch.zeros([5, 5], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 2], dtype=torch.int32),
                    torch.tensor([2, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.5, 3.5], dtype=torch.float32),
            ),
            param(
                test_name="3d_indices_float",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([1, 2], dtype=torch.int32),
                    torch.tensor([2, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([5.5, 7.5], dtype=torch.float32),
            ),
            param(
                test_name="4d_indices_float",
                source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([1, 0], dtype=torch.int32),
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([1, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([5.5, 7.5], dtype=torch.float32),
            ),
            param(
                test_name="3d_indices_float_broadcast_index",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([0, 1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([10], dtype=torch.int32),
            ),
            param(
                test_name="3d_indices_broadcast_1dim",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.int32),
                indices_tensor=(torch.tensor([1], dtype=torch.int32),),
                value_tensor=torch.tensor([7], dtype=torch.int32),
            ),
            param(
                test_name="2d_indices_broadcast_1dim",
                source_tensor=torch.zeros([4, 4], dtype=torch.int32),
                indices_tensor=(torch.tensor([1, 3], dtype=torch.int32),),
                value_tensor=torch.tensor([5], dtype=torch.int32),
            ),
            param(
                test_name="4d_indices_broadcast_2dim",
                source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.int32),
                indices_tensor=(
                    torch.tensor([0], dtype=torch.int32),
                    torch.tensor([1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([9], dtype=torch.int32),
            ),
            param(
                test_name="4d_indices_none_none_single_idx",
                source_tensor=torch.zeros([1, 2, 5, 3], dtype=torch.int32),
                indices_tensor=(None, None, torch.tensor(2, dtype=torch.int32)),
                value_tensor=torch.tensor(
                    [[[10, 20, 30], [40, 50, 60]]], dtype=torch.int32
                ),
            ),
            param(
                test_name="5d_indices_none_none_single_idx",
                source_tensor=torch.zeros((1, 2, 5, 3, 5), dtype=torch.int32),
                indices_tensor=[None, None, torch.tensor(2, dtype=torch.int32), None],
                value_tensor=torch.tensor(
                    [
                        [
                            [
                                [10, 20, 30, 40, 50],
                                [60, 70, 80, 90, 100],
                                [110, 120, 130, 140, 150],
                            ],
                            [
                                [160, 170, 180, 190, 200],
                                [210, 220, 230, 240, 250],
                                [260, 270, 280, 290, 300],
                            ],
                        ]
                    ],
                    dtype=torch.int32,
                ),
            ),
            # param(
            #     test_name="4d_indices_none_none_multiple_idx_broadcast_error",
            #     source_tensor=torch.zeros([1, 2, 5, 3], dtype=torch.float32),
            #     indices_tensor=(None, None, torch.tensor([0, 1, 2], dtype=torch.int64)),
            #     value_tensor=torch.randn([2, 3, 3], dtype=torch.float32),
            # ),
            param(
                test_name="trailing_none_after_tensor",
                # K=1: indexed dim first, trailing free dims as None
                source_tensor=torch.zeros([4, 3, 2], dtype=torch.float32),
                indices_tensor=(torch.tensor([1, 3], dtype=torch.int64), None, None),
                value_tensor=torch.ones([2, 3, 2], dtype=torch.float32),
            ),
            param(
                test_name="discontinuous_test",
                source_tensor=torch.zeros([2, 4, 4], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 0, 1], dtype=torch.int64),
                    None,
                    torch.tensor([0, 0, 1], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([2, 3, 3, 4], dtype=torch.float32),
            ),
            param(
                test_name="discontinuous_test_two",
                source_tensor=torch.zeros([2, 4, 4, 2], dtype=torch.float32),
                indices_tensor=(
                    None,
                    torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                    None,
                    torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([2, 3, 3, 4], dtype=torch.float32),
            ),
            param(
                test_name="continuous_test",
                source_tensor=torch.zeros([2, 4, 4, 2], dtype=torch.float32),
                indices_tensor=(
                    None,
                    None,
                    torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                    torch.tensor([0, 0, 1, 1], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([2, 3, 3, 4], dtype=torch.float32),
            ),
            # --- dtype coverage (mirrors PyTorch's test_index_put_src_datatype) ---
            param(
                test_name="bfloat16_1d_single",
                source_tensor=torch.zeros([5], dtype=torch.bfloat16),
                indices_tensor=(torch.tensor([0, 3], dtype=torch.int32),),
                value_tensor=torch.tensor([1.5, 3.5], dtype=torch.bfloat16),
            ),
            param(
                test_name="bfloat16_2d_multiple",
                source_tensor=torch.zeros([5, 5], dtype=torch.bfloat16),
                indices_tensor=(
                    torch.tensor([0, 2], dtype=torch.int32),
                    torch.tensor([2, 0], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.5, 3.5], dtype=torch.bfloat16),
            ),
            param(
                test_name="float16_1d_single",
                source_tensor=torch.zeros([5], dtype=torch.float16),
                indices_tensor=(torch.tensor([1, 4], dtype=torch.int32),),
                value_tensor=torch.tensor([2.0, 4.0], dtype=torch.float16),
            ),
            param(
                test_name="float16_2d_multiple",
                source_tensor=torch.zeros([4, 4], dtype=torch.float16),
                indices_tensor=(
                    torch.tensor([0, 3], dtype=torch.int32),
                    torch.tensor([1, 2], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.0, 2.0], dtype=torch.float16),
            ),
            # --- index dtype: int64 (mirrors PyTorch's test_index_ind_dtype) ---
            param(
                test_name="int64_indices_2d",
                source_tensor=torch.zeros([4, 4], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            ),
            # --- accumulate=True, unique indices (safe for TRT scatter) ---
            # Mirrors PyTorch's test_index_put_accumulate_expanded_values (no duplicates).
            param(
                test_name="accumulate_true_unique_indices_1d",
                source_tensor=torch.ones([6], dtype=torch.float32),
                indices_tensor=(torch.tensor([0, 2, 4], dtype=torch.int64),),
                value_tensor=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
                accumulate=True,
            ),
            param(
                test_name="accumulate_true_unique_indices_2d",
                source_tensor=torch.ones([4, 3], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 2], dtype=torch.int64),
                    torch.tensor([1, 2], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([5.0, 7.0], dtype=torch.float32),
                accumulate=True,
            ),
            # Broadcast: single value written to multiple unique positions.
            param(
                test_name="accumulate_true_broadcast_scalar_value",
                source_tensor=torch.zeros([5, 2], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 1, 3], dtype=torch.int64),
                    torch.tensor([0, 1, 0], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([1.0], dtype=torch.float32),
                accumulate=True,
            ),
            # --- accumulate=True with duplicate indices (uses _index_put_scatter_add) ---
            # These exercise the matmul-based scatter_add path which handles
            # duplicate positions correctly (mirrors test_index_put_accumulate_duplicate_indices).
            param(
                test_name="1d_duplicate_indices_accumulate",
                source_tensor=torch.zeros([6], dtype=torch.float32),
                indices_tensor=(torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64),),
                value_tensor=torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32
                ),
                accumulate=True,
            ),
            param(
                test_name="2d_indices_accumulate_True",
                source_tensor=torch.zeros([5, 5], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 0], dtype=torch.int32),
                    torch.tensor([1, 1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.0, 2.0], dtype=torch.float32),
                accumulate=True,
            ),
            param(
                test_name="3d_indices_accumulate_True",
                source_tensor=torch.zeros([3, 3, 3], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 0], dtype=torch.int32),
                    torch.tensor([1, 1], dtype=torch.int32),
                    torch.tensor([2, 2], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.0, 2.0], dtype=torch.float32),
                accumulate=True,
            ),
            param(
                test_name="4d_indices_accumulate_True",
                source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 0], dtype=torch.int32),
                    torch.tensor([1, 1], dtype=torch.int32),
                    torch.tensor([0, 0], dtype=torch.int32),
                    torch.tensor([1, 1], dtype=torch.int32),
                ),
                value_tensor=torch.tensor([1.0, 2.0], dtype=torch.float32),
                accumulate=True,
            ),
            # Negative indices with accumulate (mirrors test_index_put_accumulate_large_tensor).
            param(
                test_name="accumulate_negative_indices",
                source_tensor=torch.zeros([6], dtype=torch.float32),
                indices_tensor=(torch.tensor([-1, -1, -3], dtype=torch.int64),),
                value_tensor=torch.tensor([5.0, 7.0, 3.0], dtype=torch.float32),
                accumulate=True,
            ),
            # bfloat16 + duplicate indices: computation stays in bfloat16 (no forced fp32 cast).
            param(
                test_name="accumulate_bfloat16_duplicate",
                source_tensor=torch.zeros([4, 4], dtype=torch.bfloat16),
                indices_tensor=(
                    torch.tensor([0, 0, 2], dtype=torch.int64),
                    torch.tensor([1, 1, 3], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16),
                accumulate=True,
            ),
            # float16 + duplicate indices.
            param(
                test_name="accumulate_float16_duplicate",
                source_tensor=torch.zeros([4, 4], dtype=torch.float16),
                indices_tensor=(
                    torch.tensor([1, 1, 3], dtype=torch.int64),
                    torch.tensor([0, 0, 2], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([2.0, 3.0, 5.0], dtype=torch.float16),
                accumulate=True,
            ),
            # Partial broadcast: one index covers a single position on dim-1 while
            # dim-0 has multiple positions — mirrors test_index_put_accumulate_expanded_values
            # (t[tensor([0,1,2,3]), tensor([1])] += 1.0).
            param(
                test_name="accumulate_partial_dim1_broadcast",
                source_tensor=torch.zeros([5, 2], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                    torch.tensor([1], dtype=torch.int64),
                ),
                value_tensor=torch.tensor([1.0], dtype=torch.float32),
                accumulate=True,
            ),
        ]
    )
    def test_index_put(
        self, test_name, source_tensor, indices_tensor, value_tensor, accumulate=False
    ):
        if accumulate and ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("ScatterAdd plugin not available in TRT RTX")

        @torch._dynamo.assume_constant_result
        def get_indices_tensor():
            return indices_tensor

        class TestIndexPut(torch.nn.Module):
            def forward(self, source_tensor, value_tensor):
                indices_tensor_const = get_indices_tensor()
                return torch.ops.aten.index_put.default(
                    source_tensor, indices_tensor_const, value_tensor, accumulate
                )

        self.run_test(
            TestIndexPut(),
            inputs=[source_tensor, value_tensor],
            enable_passes=True,
            use_dynamo_tracer=True,
            use_explicit_typing=True,
        )

    def test_index_add_dynamic_shape(self):

        class Model(torch.nn.Module):
            def forward(self, x, y, z, a, b):
                x.index_add_(0, y, z)
                x.index_add_(0, a, b)
                return x

        dim = 10
        model = Model().cuda()
        inputs = [
            torch.ones((12, dim)).half().cuda(),
            torch.tensor([0, 1]).cuda(),
            torch.randn((2, dim)).half().cuda(),
            torch.tensor([2, 9, 11]).cuda(),
            torch.randn((3, dim)).half().cuda(),
        ]
        torch_output = model.cuda().forward(*inputs)
        seq_len1 = torch.export.Dim("seq_len1", min=1, max=128)
        seq_len2 = torch.export.Dim("seq_len2", min=1, max=128)
        seq_len3 = torch.export.Dim("seq_len3", min=1, max=128)

        ep = torch.export.export(
            model,
            tuple(inputs),
            dynamic_shapes=(
                {0: seq_len1},
                {0: seq_len2},
                {0: seq_len2},
                {0: seq_len3},
                {0: seq_len3},
            ),
        )

        trt_mod = torchtrt.dynamo.compile(
            ep,
            inputs,
            enabled_precisions={torch.float16},
            min_block_size=1,
            use_explicit_typing=False,
            use_fp32_acc=False,
            disable_tf32=True,
        )
        result = trt_mod(*inputs)
        assert torch.allclose(result, torch_output, atol=1e-4, rtol=1e-4)

    def test_bool_mask_test(self):

        source_tensor = torch.ones([5, 10], dtype=torch.float32).cuda()
        indices_tensor = torch.tensor([False, False, True, False, True])
        value_tensor = torch.zeros([2, 10], dtype=torch.float32).cuda()

        dim1 = torch.export.Dim("dim1", min=1, max=5)
        dim2 = torch.export.Dim("dim2", min=1, max=5)

        class TestIndexPut(torch.nn.Module):
            def forward(self, source_tensor, indices_tensor, value_tensor):
                source_tensor[indices_tensor] = value_tensor
                return source_tensor

        model = TestIndexPut()
        torch_output = model.forward(source_tensor, indices_tensor, value_tensor)

        ep = torch.export.export(
            model,
            (source_tensor, indices_tensor, value_tensor),
            dynamic_shapes=({0: dim1}, {0: dim1}, {0: dim2}),
        )
        trt_engine = torchtrt.dynamo.compile(
            ep,
            inputs=(source_tensor, indices_tensor, value_tensor),
            enabled_precisions={torch.float32},
            min_block_size=1,
            use_explicit_typing=False,
            use_fp32_acc=False,
            disable_tf32=True,
            use_python_runtime=True,
        )
        result = trt_engine(source_tensor, indices_tensor, value_tensor)

        torch.allclose(result, torch_output, atol=1e-4, rtol=1e-4)

    def test_index_put_dynamic_index_length(self):
        """index_put where the index tensor itself has a dynamic length (N dynamic).

        Pattern: src[idx] = values  — no free dims, K=rank=1, index length dynamic.
        """

        class IndexPutDynN(torch.nn.Module):
            def forward(self, src, values, idx):
                return torch.ops.aten.index_put.default(src, [idx], values)

        src = torch.zeros(16, dtype=torch.float32, device="cuda")
        n_dim = torch.export.Dim("n", min=1, max=16)

        model = IndexPutDynN().eval().cuda()
        # concrete inputs for reference
        idx = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        values = torch.ones(3, dtype=torch.float32, device="cuda")
        torch_output = model(src.clone(), values, idx)

        ep = torch.export.export(
            model,
            args=(src, values, idx),
            dynamic_shapes={"src": {}, "values": {0: n_dim}, "idx": {0: n_dim}},
        )
        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(shape=(16,), dtype=torch.float32),
                torchtrt.Input(
                    min_shape=(1,), opt_shape=(3,), max_shape=(16,), dtype=torch.float32
                ),
                torchtrt.Input(
                    min_shape=(1,), opt_shape=(3,), max_shape=(16,), dtype=torch.int32
                ),
            ],
            min_block_size=1,
        )
        result = trt_mod(src.clone(), values, idx)
        assert torch.allclose(
            result, torch_output, atol=1e-4, rtol=1e-4
        ), f"Dynamic index length mismatch: max diff = {(result - torch_output).abs().max()}"

    def test_kv_cache_dynamic_batch(self):
        """index_put with a dynamic free dimension (batch) — issue #4139.

        Pattern: cache[..., idx, :] = values  where dim-1 (batch) is dynamic
        and dim-2 (cache/time) is the indexed static dimension.
        """

        class KVCacheModel(torch.nn.Module):
            def forward(self, cache, values, idx):
                cache[..., idx, :] = values
                return cache

        N = 4
        max_ctx = 256
        L = 1
        H = 512

        cache = torch.zeros(2, N, max_ctx, H, dtype=torch.float16, device="cuda")
        values = torch.randn(2, N, L, H, dtype=torch.float16, device="cuda")
        idx = torch.tensor([3], dtype=torch.long, device="cuda")

        model = KVCacheModel().eval().cuda()
        torch_output = model(cache.clone(), values, idx)

        batch_dim = torch.export.Dim("batch", min=1, max=64)
        ep = torch.export.export(
            model,
            args=(cache, values, idx),
            dynamic_shapes={
                "cache": {1: batch_dim},
                "values": {1: batch_dim},
                "idx": {},
            },
        )

        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(
                    min_shape=(2, 1, max_ctx, H),
                    opt_shape=(2, N, max_ctx, H),
                    max_shape=(2, 64, max_ctx, H),
                    dtype=torch.float16,
                ),
                torchtrt.Input(
                    min_shape=(2, 1, L, H),
                    opt_shape=(2, N, L, H),
                    max_shape=(2, 64, L, H),
                    dtype=torch.float16,
                ),
                torchtrt.Input(
                    min_shape=(L,),
                    opt_shape=(L,),
                    max_shape=(L,),
                    dtype=torch.long,
                ),
            ],
            use_explicit_typing=True,
            min_block_size=1,
        )

        result = trt_mod(cache.clone(), values, idx)
        assert torch.allclose(
            result, torch_output, atol=1e-3, rtol=1e-3
        ), f"KV-cache index_put mismatch: max diff = {(result - torch_output).abs().max()}"

    def test_accumulate_random_walk_duplicate_indices(self):
        """accumulate=True on 1-D input where indices are generated by a random walk
        (many duplicates interleaved).  Mirrors PyTorch's
        test_index_put_accumulate_duplicate_indices, scaled to a TRT-friendly size.
        """
        import random

        torch.manual_seed(42)
        random.seed(42)

        class AccumDup(torch.nn.Module):
            def forward(self, src, values, idx):
                return torch.ops.aten.index_put.default(src, [idx], values, True)

        for trial in range(5):
            n = random.randint(8, 32)
            delta = torch.empty(n, dtype=torch.float32).uniform_(-1, 1)
            idx = delta.cumsum(0).long()
            src_size = int(idx.abs().max().item()) + 1
            src = torch.randn(src_size, dtype=torch.float32, device="cuda")
            values = torch.randn(n, dtype=torch.float32, device="cuda")
            idx_cuda = idx.cuda()

            model = AccumDup().eval().cuda()
            torch_output = model(src.clone(), values, idx_cuda)

            ep = torch.export.export(model, args=(src, values, idx_cuda))
            trt_mod = torchtrt.dynamo.compile(
                ep,
                arg_inputs=[
                    torchtrt.Input(shape=(src_size,), dtype=torch.float32),
                    torchtrt.Input(shape=(n,), dtype=torch.float32),
                    torchtrt.Input(shape=(n,), dtype=torch.int64),
                ],
                min_block_size=1,
            )
            result = trt_mod(src.clone(), values, idx_cuda)
            assert torch.allclose(result, torch_output, atol=1e-4, rtol=1e-4), (
                f"Trial {trial}: random-walk accumulate mismatch, "
                f"max diff = {(result - torch_output).abs().max()}"
            )

    def test_accumulate_expanded_values_3d(self):
        """accumulate=True with 3-D source and broadcast mesh indices — mirrors the
        second half of PyTorch's test_index_put_accumulate_expanded_values.

        Pattern: t[tensor([0]), arange(3)[:,None], arange(2)[None,:]] += values
        """

        class AccumMesh(torch.nn.Module):
            def forward(self, src, values, i0, i1, i2):
                return torch.ops.aten.index_put.default(src, [i0, i1, i2], values, True)

        src = torch.zeros(4, 3, 2, dtype=torch.float32, device="cuda")
        i0 = torch.tensor([0], dtype=torch.int64, device="cuda")
        i1 = torch.arange(3, dtype=torch.int64, device="cuda").view(3, 1)
        i2 = torch.arange(2, dtype=torch.int64, device="cuda").view(1, 2)
        values = torch.tensor([-1.0, -2.0], dtype=torch.float32, device="cuda")

        model = AccumMesh().eval().cuda()
        torch_output = model(src.clone(), values, i0, i1, i2)

        ep = torch.export.export(model, args=(src, values, i0, i1, i2))
        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(shape=(4, 3, 2), dtype=torch.float32),
                torchtrt.Input(shape=(2,), dtype=torch.float32),
                torchtrt.Input(shape=(1,), dtype=torch.int64),
                torchtrt.Input(shape=(3, 1), dtype=torch.int64),
                torchtrt.Input(shape=(1, 2), dtype=torch.int64),
            ],
            min_block_size=1,
        )
        result = trt_mod(src.clone(), values, i0, i1, i2)
        assert torch.allclose(
            result, torch_output, atol=1e-4, rtol=1e-4
        ), f"3D expand accumulate mismatch: max diff = {(result - torch_output).abs().max()}"

    def test_empty_index_no_op(self):
        """index_put with an empty index tensor is a no-op — output equals input.

        Mirrors PyTorch's test_empty_index: x[empty_idx] = values leaves x unchanged.
        Uses static shapes (torch.export rejects Dim(min=0) with a concrete 0-size tensor).
        """

        class EmptyIndexPut(torch.nn.Module):
            def forward(self, src, values, idx):
                return torch.ops.aten.index_put.default(src, [idx], values)

        src = torch.arange(8, dtype=torch.float32, device="cuda")
        idx = torch.tensor([], dtype=torch.int64, device="cuda")
        values = torch.tensor([], dtype=torch.float32, device="cuda")

        model = EmptyIndexPut().eval().cuda()
        torch_output = model(src.clone(), values, idx)

        ep = torch.export.export(model, args=(src, values, idx))
        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(shape=(8,), dtype=torch.float32),
                torchtrt.Input(shape=(0,), dtype=torch.float32),
                torchtrt.Input(shape=(0,), dtype=torch.int64),
            ],
            min_block_size=1,
        )
        result = trt_mod(src.clone(), values, idx)
        assert torch.allclose(
            result, torch_output, atol=1e-4, rtol=1e-4
        ), f"Empty-index no-op mismatch: {result} vs {torch_output}"

    def test_index_ind_dtype_int_vs_long(self):
        """int32 and int64 index tensors must produce identical results.

        Mirrors PyTorch's test_index_ind_dtype.
        """

        class IndexPutIntIdx(torch.nn.Module):
            def forward(self, src, values, idx):
                return torch.ops.aten.index_put.default(src, [idx], values)

        src = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")

        idx_long = torch.arange(4, dtype=torch.int64, device="cuda")
        idx_int = idx_long.to(torch.int32)

        model = IndexPutIntIdx().eval().cuda()
        ref_long = model(src.clone(), values, idx_long)
        ref_int = model(src.clone(), values, idx_int)
        assert torch.allclose(ref_long, ref_int), "CPU int32 vs int64 mismatch"

        ep_long = torch.export.export(model, args=(src, values, idx_long))
        ep_int = torch.export.export(model, args=(src, values, idx_int))

        trt_long = torchtrt.dynamo.compile(
            ep_long,
            arg_inputs=[
                torchtrt.Input(shape=(4, 4), dtype=torch.float32),
                torchtrt.Input(shape=(4,), dtype=torch.float32),
                torchtrt.Input(shape=(4,), dtype=torch.int64),
            ],
            min_block_size=1,
        )
        trt_int = torchtrt.dynamo.compile(
            ep_int,
            arg_inputs=[
                torchtrt.Input(shape=(4, 4), dtype=torch.float32),
                torchtrt.Input(shape=(4,), dtype=torch.float32),
                torchtrt.Input(shape=(4,), dtype=torch.int32),
            ],
            min_block_size=1,
        )

        out_long = trt_long(src.clone(), values, idx_long)
        out_int = trt_int(src.clone(), values, idx_int)

        assert torch.allclose(
            out_long, ref_long, atol=1e-4, rtol=1e-4
        ), "TRT int64 mismatch"
        assert torch.allclose(
            out_int, ref_int, atol=1e-4, rtol=1e-4
        ), "TRT int32 mismatch"
        assert torch.allclose(
            out_long, out_int, atol=1e-4, rtol=1e-4
        ), "TRT int32 vs int64 inconsistency"

    def test_accumulate_non_contiguous_source(self):
        """accumulate=True on a non-contiguous (sliced) source tensor.

        Mirrors PyTorch's test_index_put_accumulate_non_contiguous.
        Uses unique indices so TRT scatter is correct.
        """

        class AccumNonContig(torch.nn.Module):
            def forward(self, src, values, idx):
                # src is already a non-contiguous slice passed in
                return torch.ops.aten.index_put.default(src, [idx], values, True)

        base = torch.zeros(5, 2, 2, dtype=torch.float32, device="cuda")
        # take a non-contiguous slice: shape (5, 2), stride (4, 1)
        src_slice = base[:, 0, :]
        assert not src_slice.is_contiguous()

        idx = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
        values = torch.ones(2, 2, dtype=torch.float32, device="cuda")

        model = AccumNonContig().eval().cuda()
        torch_output = model(src_slice.clone(), values, idx)

        ep = torch.export.export(
            model,
            args=(src_slice.contiguous(), values, idx),
        )
        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(shape=(5, 2), dtype=torch.float32),
                torchtrt.Input(shape=(2, 2), dtype=torch.float32),
                torchtrt.Input(shape=(2,), dtype=torch.int64),
            ],
            min_block_size=1,
        )
        result = trt_mod(src_slice.contiguous(), values, idx)
        assert torch.allclose(
            result, torch_output, atol=1e-4, rtol=1e-4
        ), f"Non-contiguous accumulate mismatch: max diff = {(result - torch_output).abs().max()}"

    def test_accumulate_expanded_values_broadcast(self):
        """accumulate=True with value broadcasting — 0D scalar and 1D values
        broadcast across unique indexed positions.

        Mirrors PyTorch's test_index_put_accumulate_expanded_values (unique indices only).
        """

        class AccumBroadcast(torch.nn.Module):
            def forward(self, src, values, idx0, idx1):
                return torch.ops.aten.index_put.default(src, [idx0, idx1], values, True)

        src = torch.zeros(5, 2, dtype=torch.float32, device="cuda")
        idx0 = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        idx1 = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
        values_1d = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        model = AccumBroadcast().eval().cuda()
        torch_output = model(src.clone(), values_1d, idx0, idx1)

        ep = torch.export.export(model, args=(src, values_1d, idx0, idx1))
        trt_mod = torchtrt.dynamo.compile(
            ep,
            arg_inputs=[
                torchtrt.Input(shape=(5, 2), dtype=torch.float32),
                torchtrt.Input(shape=(1,), dtype=torch.float32),
                torchtrt.Input(shape=(4,), dtype=torch.int64),
                torchtrt.Input(shape=(4,), dtype=torch.int64),
            ],
            min_block_size=1,
        )
        result = trt_mod(src.clone(), values_1d, idx0, idx1)
        assert torch.allclose(
            result, torch_output, atol=1e-4, rtol=1e-4
        ), f"Accumulate broadcast mismatch: max diff = {(result - torch_output).abs().max()}"

    # ------------------------------------------------------------------
    # Duplicate-index tests for realistic use-case models
    # These mirror the scenarios in experiments/bench_index_put_scatter_add.py
    # and verify that _index_put_scatter_add correctly accumulates into
    # duplicate positions when index_put is embedded in a larger graph.
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        ENABLED_FEATURES.tensorrt_rtx,
        reason="ScatterAdd plugin not available in TRT RTX",
    )
    def test_kv_cache_duplicate_slot_writes(self):
        """KV-cache style: linear projection → index_put(accumulate=True) into
        a flat cache with duplicate slot indices → output projection.

        Multiple writes to the same cache slot must sum, not overwrite.
        """
        N, S, D = 6, 16, 32

        # Positions with duplicates: slots 2 and 5 are written twice
        positions = torch.tensor([0, 2, 2, 5, 5, 7], dtype=torch.int64)

        @torch._dynamo.assume_constant_result
        def get_positions():
            return positions

        class KVCacheDupWrites(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = torch.nn.Linear(D, D, bias=False)
                self.proj_out = torch.nn.Linear(D, D, bias=False)

            def forward(self, tokens, cache):
                # tokens: (N, D), cache: (S, D)
                feats = self.proj_in(tokens)
                cache = cache.index_put((get_positions(),), feats, accumulate=True)
                return self.proj_out(cache)

        tokens = torch.randn(N, D)
        cache = torch.zeros(S, D)

        self.run_test(
            KVCacheDupWrites().cuda(),
            inputs=[tokens, cache],
            use_dynamo_tracer=True,
            enable_passes=True,
            use_explicit_typing=True,
        )

    @pytest.mark.skipif(
        ENABLED_FEATURES.tensorrt_rtx,
        reason="ScatterAdd plugin not available in TRT RTX",
    )
    def test_sparse_embedding_duplicate_seq_ids(self):
        """Sparse embedding accumulation: embedding lookup → index_put(accumulate=True)
        into per-sequence accumulators where many tokens map to the same sequence → ReLU.

        Multiple tokens per sequence must be summed, not overwritten.
        """
        B, N = 4, 20

        # Many tokens map to the same sequence: each seq gets ~5 tokens
        seq_ids = torch.tensor(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            dtype=torch.int64,
        )

        @torch._dynamo.assume_constant_result
        def get_seq_ids():
            return seq_ids

        class SparseEmbedAccum(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 16)
                self.head = torch.nn.Linear(16, 8, bias=False)

            def forward(self, token_ids, accum):
                # token_ids: (N,) int64, accum: (B, 16)
                embs = self.embedding(token_ids)
                accum = accum.index_put((get_seq_ids(),), embs, accumulate=True)
                return self.head(torch.relu(accum))

        token_ids = torch.randint(0, 64, (N,))
        accum = torch.zeros(B, 16)

        self.run_test(
            SparseEmbedAccum().cuda(),
            inputs=[token_ids, accum],
            use_dynamo_tracer=True,
            enable_passes=True,
            use_explicit_typing=True,
        )

    @pytest.mark.skipif(
        ENABLED_FEATURES.tensorrt_rtx,
        reason="ScatterAdd plugin not available in TRT RTX",
    )
    def test_histogram_conv_duplicate_bin_ids(self):
        """Histogram accumulation: Conv1d → index_put(accumulate=True) into histogram
        bins where many frames land in the same bin → mean pool → linear.

        Multiple frames writing to the same bin must accumulate, not overwrite.
        """
        C, L, n_bins = 4, 12, 6

        # Skewed bin assignment: bins 0 and 1 receive many frames
        bin_ids = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5],
            dtype=torch.int64,
        )

        @torch._dynamo.assume_constant_result
        def get_bin_ids():
            return bin_ids

        class HistConvAccum(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(C, 8, kernel_size=3, padding=1)
                self.head = torch.nn.Linear(8, 4, bias=False)

            def forward(self, signal, hist):
                # signal: (1, C, L), hist: (n_bins, 8)
                feat = self.conv(signal).squeeze(0).T  # (L, 8)
                hist = hist.index_put((get_bin_ids(),), feat, accumulate=True)
                return self.head(hist.mean(dim=0))

        signal = torch.randn(1, C, L)
        hist = torch.zeros(n_bins, 8)

        self.run_test(
            HistConvAccum().cuda(),
            inputs=[signal, hist],
            use_dynamo_tracer=True,
            enable_passes=True,
            use_explicit_typing=True,
        )


if __name__ == "__main__":
    run_tests()
