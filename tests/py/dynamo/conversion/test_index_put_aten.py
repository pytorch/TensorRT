import torch
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


if __name__ == "__main__":
    run_tests()
