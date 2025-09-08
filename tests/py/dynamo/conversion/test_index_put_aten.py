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
            param(
                test_name="4d_indices_none_none_multiple_idx_broadcast_error",
                source_tensor=torch.zeros([1, 2, 5, 3], dtype=torch.float32),
                indices_tensor=(None, None, torch.tensor([0, 1, 2], dtype=torch.int64)),
                value_tensor=torch.randn([2, 3, 3], dtype=torch.float32),
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
        with torchtrt.dynamo.Debugger(
            log_level="debug",
            capture_fx_graph_after=["remove_num_users_is_0_nodes"],
            logging_dir="/home/profile/logging/moe",
            engine_builder_monitor=False,
        ):
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


if __name__ == "__main__":
    run_tests()
