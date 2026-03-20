import torch
import torch_tensorrt as torchtrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


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
            # param(
            #     test_name="2d_indices_accumulate_True",
            #     source_tensor=torch.zeros([5, 5], dtype=torch.int32),
            #     indices_tensor=(torch.tensor([0, 0], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)),
            #     value_tensor=torch.tensor([1, 2], dtype=torch.int32),
            #     accumulate=True,
            # ),
            # param(
            #     test_name="3d_indices_accumulate_True",
            #     source_tensor=torch.zeros([3, 3, 3], dtype=torch.int32),
            #     indices_tensor=(torch.tensor([0, 0], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32), torch.tensor([2, 2], dtype=torch.int32)),
            #     value_tensor=torch.tensor([1, 2], dtype=torch.int32),
            #     accumulate=True,
            # ),
            # param(
            #     test_name="4d_indices_accumulate_True",
            #     source_tensor=torch.zeros([2, 2, 2, 2], dtype=torch.int32),
            #     indices_tensor=(torch.tensor([0, 0], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)),
            #     value_tensor=torch.tensor([1, 2], dtype=torch.int32),
            #     accumulate=True,
            # ),
        ]
    )
    def test_index_put(
        self, test_name, source_tensor, indices_tensor, value_tensor, accumulate=False
    ):
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
                torchtrt.Input(min_shape=(1,), opt_shape=(3,), max_shape=(16,), dtype=torch.float32),
                torchtrt.Input(min_shape=(1,), opt_shape=(3,), max_shape=(16,), dtype=torch.int32),
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


if __name__ == "__main__":
    run_tests()
