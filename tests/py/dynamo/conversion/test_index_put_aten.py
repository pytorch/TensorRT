# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch_tensorrt as torchtrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import ENABLED_FEATURES

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


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
            # bfloat16 + duplicate indices.
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
            # A trailing None selects every column, so ScatterAdd still takes it.
            param(
                test_name="accumulate_trailing_none",
                source_tensor=torch.zeros([6, 3], dtype=torch.float32),
                indices_tensor=(torch.tensor([1, 3, 3], dtype=torch.int64), None),
                value_tensor=torch.ones([3, 3], dtype=torch.float32),
                accumulate=True,
            ),
            # Values broadcast to input[indices] as in PyTorch, even where one of their
            # dims happens to have the index length.
            param(
                test_name="values_dim_equal_to_index_length",
                source_tensor=torch.zeros([5, 3, 4], dtype=torch.float32),
                indices_tensor=(torch.tensor([4, 0, 3, 1]),),
                value_tensor=torch.arange(4, dtype=torch.float32),
            ),
            param(
                test_name="values_without_leading_free_dim",
                source_tensor=torch.zeros([4, 4, 3], dtype=torch.float32),
                indices_tensor=(None, torch.tensor([3, 1, 0, 2])),
                value_tensor=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            ),
            param(
                test_name="2d_index_before_free_dim",
                source_tensor=torch.zeros([6, 4], dtype=torch.float32),
                indices_tensor=(torch.tensor([[0, 2, 5], [4, 1, 3]]),),
                value_tensor=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            ),
            param(
                test_name="0d_index_after_none",
                source_tensor=torch.zeros([3, 4], dtype=torch.float32),
                indices_tensor=(None, torch.tensor(2)),
                value_tensor=torch.arange(3, dtype=torch.float32),
            ),
            param(
                test_name="2d_bool_mask_after_none",
                source_tensor=torch.zeros([2, 3, 4], dtype=torch.float32),
                indices_tensor=(
                    None,
                    torch.tensor(
                        [
                            [True, False, False, True],
                            [False, True, False, False],
                            [False, False, True, True],
                        ]
                    ),
                ),
                value_tensor=torch.arange(5, dtype=torch.float32),
            ),
            param(
                test_name="mask_between_nones_after_0d_index",
                source_tensor=torch.zeros([2, 3, 2, 1], dtype=torch.float32),
                indices_tensor=(
                    torch.tensor(0),
                    None,
                    torch.tensor([[True], [False]]),
                    None,
                ),
                value_tensor=torch.tensor(7.0),
            ),
            param(
                test_name="index_after_mask_and_none_past_last_dim",
                source_tensor=torch.zeros([2, 1, 1], dtype=torch.float32),
                indices_tensor=(torch.tensor([[True], [False]]), None, torch.tensor(0)),
                value_tensor=torch.tensor(7.0),
            ),
        ]
    )
    def test_index_put(
        self, test_name, source_tensor, indices_tensor, value_tensor, accumulate=False
    ):
        if accumulate and ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("ScatterAdd plugin not available in TRT RTX")
        if ENABLED_FEATURES.tensorrt_rtx and any(
            index is not None and index.dtype == torch.bool for index in indices_tensor
        ):
            pytest.skip("Skipped on tensorrt_rtx due to nonzero not supported")

        # Keyed on the case's dtype rather than its index: the generated ids are bare
        # test_index_put_<N>, and there is a commented-out param above, so any index-based
        # list would drift the moment a case is added or restored.
        if torch.bfloat16 in (source_tensor.dtype, value_tensor.dtype):
            skip_if_trt_rtx_turing(self, "bfloat16")

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
            min_block_size=1,
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
            min_block_size=1,
            use_fp32_acc=False,
            disable_tf32=True,
        )
        result = trt_engine(source_tensor, indices_tensor, value_tensor)

        torch.allclose(result, torch_output, atol=1e-4, rtol=1e-4)

    def test_bool_mask_with_data_dependent_count(self):
        # The True count is only known at runtime, while values has a static row count.
        class AddRows(torch.nn.Module):
            def forward(self, data, mask, values):
                data = data.clone()
                data[mask, :] += values  # Qwen3-VL adds visual embeddings this way
                return data

        class SetRows(torch.nn.Module):
            def forward(self, data, mask, values):
                data = data.clone()
                data[mask] = values
                return data

        class SetRowsAtColumns(torch.nn.Module):
            def forward(self, data, mask, values):
                data = data.clone()
                data[mask, torch.tensor([0, 2, 4, 5], device=data.device)] = values
                return data

        values = torch.randn(4, 8, device="cuda")
        mask = torch.zeros(1, 10, dtype=torch.bool, device="cuda")
        mask[0, 3:7] = True
        one_true = torch.zeros(10, dtype=torch.bool, device="cuda")
        one_true[5] = True
        cases = [
            (AddRows(), torch.randn(1, 10, 8, device="cuda"), mask, ()),
            (SetRows(), torch.randn(10, 8, device="cuda"), mask[0], ()),
            # One True entry broadcasts across the four column indices.
            (
                SetRowsAtColumns(),
                torch.randn(10, 6, 8, device="cuda"),
                mask[0],
                (one_true,),
            ),
        ]
        for model, data, case_mask, other_masks in cases:
            with self.subTest(type(model).__name__):
                inputs = (data, case_mask, values)
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    pass_through_build_failures=True,
                )
                for run_mask in (case_mask, *other_masks):
                    torch.testing.assert_close(
                        compiled(data, run_mask, values), model(data, run_mask, values)
                    )
                # A True count that differs from the value rows fails, as in PyTorch,
                # which raises IndexError when the op runs outside TensorRT.
                fewer = case_mask.clone()
                fewer[..., 6] = False
                with self.assertRaises((RuntimeError, IndexError)):
                    compiled(data, fewer, values)

    def test_runtime_length_one_broadcasts(self):
        # A length known only at runtime that turns out to be 1 repeats, as in PyTorch.
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("Skipped on tensorrt_rtx due to nonzero not supported")

        class SetRows(torch.nn.Module):
            def forward(self, data, mask, values):
                index = mask.nonzero().squeeze(1)
                return torch.ops.aten.index_put.default(
                    data, [index], values[values > 0]
                )

        class SetColumns(torch.nn.Module):
            def forward(self, data, mask, values):
                index = mask.nonzero().squeeze(1)
                column = torch.tensor([1], device=data.device)
                return torch.ops.aten.index_put.default(
                    data, [None, index, column], values[values > 0]
                )

        class SetPlanes(torch.nn.Module):
            def forward(self, data, mask, values):
                index = mask.nonzero().squeeze(1)
                return torch.ops.aten.index_put.default(data, [None, index], values)

        class SetPairs(torch.nn.Module):
            def forward(self, data, rows, columns, values):
                return torch.ops.aten.index_put.default(
                    data,
                    [rows.nonzero().squeeze(1), columns.nonzero().squeeze(1)],
                    values,
                )

        # Two indices and one value, then no index, run on one engine.
        mask = torch.tensor([1, 0, 1, 0], device="cuda")
        no_mask = torch.zeros_like(mask)
        values = torch.tensor([7.0, -1.0], device="cuda")
        data_1d = torch.zeros(4, device="cuda")
        data_3d = torch.zeros(3, 4, 5, device="cuda")
        scalar = torch.tensor(7.0, device="cuda")
        cases = [
            (SetRows(), [(data_1d, mask, values), (data_1d, no_mask, values)]),
            (SetColumns(), [(data_3d, mask, values), (data_3d, no_mask, values)]),
            (SetPlanes(), [(data_3d, mask, scalar), (data_3d, no_mask, scalar)]),
            # One row and two columns broadcast to two pairs.
            (
                SetPairs(),
                [
                    (
                        torch.zeros(4, 5, device="cuda"),
                        torch.tensor([0, 0, 1, 0], device="cuda"),
                        torch.tensor([1, 0, 0, 1, 0], device="cuda"),
                        torch.tensor([3.0, 4.0], device="cuda"),
                    )
                ],
            ),
        ]
        for case, (model, runs) in enumerate(cases):
            with self.subTest(type(model).__name__, case=case):
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(model, runs[0]),
                    arg_inputs=runs[0],
                    min_block_size=1,
                    pass_through_build_failures=True,
                )
                left = [
                    node
                    for module in compiled.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                    if node.target is torch.ops.aten.index_put.default
                ]
                self.assertEqual(left, [])
                for inputs in runs:
                    torch.testing.assert_close(compiled(*inputs), model(*inputs))

        # Values with no elements write nothing, and raise as in PyTorch once the index
        # selects something.
        empty = (
            torch.zeros(3, 4, device="cuda"),
            no_mask,
            torch.zeros(3, 0, device="cuda"),
        )
        compiled = torchtrt.dynamo.compile(
            torch.export.export(SetPlanes(), empty),
            arg_inputs=empty,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        torch.testing.assert_close(compiled(*empty), SetPlanes()(*empty))
        with self.assertRaises((RuntimeError, IndexError)):
            compiled(empty[0], mask, empty[2])

        # Two values for one selected column raise as in PyTorch rather than lose one.
        compiled = torchtrt.dynamo.compile(
            torch.export.export(SetColumns(), (data_3d, mask, values)),
            arg_inputs=(data_3d, mask, values),
            min_block_size=1,
            pass_through_build_failures=True,
        )
        one_column = torch.tensor([1, 0, 0, 0], device="cuda")
        with self.assertRaises((RuntimeError, IndexError)):
            compiled(data_3d, one_column, torch.tensor([5.0, 6.0], device="cuda"))

    def test_computed_0d_index_and_value(self):
        # A computed 0-d index or value, such as value + 1, must not gain a dim, or the
        # values would move.
        class AddOne(torch.nn.Module):
            def forward(self, data, index, value):
                return torch.ops.aten.index_put.default(data, [index], value + 1)

        class NextColumn(torch.nn.Module):
            def forward(self, data, index, values):
                return torch.ops.aten.index_put.default(data, [None, index + 1], values)

        class Gate(torch.nn.Module):
            def forward(self, data, index):
                return torch.ops.aten.index_put.default(
                    data, [index > 0], torch.tensor(7.0)
                )

        index = torch.tensor(1, device="cuda")
        cases = [
            (
                AddOne(),
                (
                    torch.arange(4.0, device="cuda"),
                    index,
                    torch.tensor(7.0, device="cuda"),
                ),
            ),
            (
                NextColumn(),
                (
                    torch.zeros(3, 4, 5, device="cuda"),
                    index,
                    torch.arange(15.0, device="cuda").reshape(3, 5),
                ),
            ),
            # A 0-d mask from a comparison stays a mask.
            (Gate(), (torch.arange(4.0, device="cuda").reshape(1, 4), index)),
        ]
        for model, inputs in cases:
            with self.subTest(type(model).__name__):
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    pass_through_build_failures=True,
                )
                torch.testing.assert_close(compiled(*inputs), model(*inputs))

    def test_0d_mask_fills_everything_or_nothing(self):
        # PyTorch runs index_put with a 0-d mask as masked_fill, so the value goes to every
        # element when the mask is True and to none when it is False.
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("Skipped on tensorrt_rtx due to nonzero not supported")

        class PutWhenPositive(torch.nn.Module):
            def forward(self, data, index):
                return torch.ops.aten.index_put.default(
                    data, [None, index > 0], torch.tensor(7.0)
                )

        # PyTorch fails inside index_put for values of more than one element, so these
        # compare with input[mask] = values.
        class PutRow(torch.nn.Module):
            def forward(self, data, index, values):
                return torch.ops.aten.index_put.default(data, [index > 0], values)

        data = torch.arange(6.0, device="cuda").reshape(2, 3)
        row = torch.arange(10.0, 13.0, device="cuda")
        positive = torch.tensor(1, device="cuda")
        cases = [
            (PutWhenPositive(), (data, positive), None),
            (PutRow(), (data, positive, row), row.expand(2, 3)),
        ]
        for model, inputs, expected in cases:
            with self.subTest(f"{type(model).__name__}_{inputs[1].item()}"):
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    pass_through_build_failures=True,
                )
                if expected is None:
                    expected = model(*inputs)
                torch.testing.assert_close(compiled(*inputs), expected)

        # Values that do not broadcast to the input shape are rejected.
        for case, inputs in {
            "more_dims": (data, positive, torch.zeros(1, 2, 3, device="cuda")),
            "more_rows": (data[:1], positive, torch.zeros(2, 3, device="cuda")),
            "longer_row": (data[:, :1], positive, row),
        }.items():
            with self.subTest(case):
                with self.assertRaisesRegex(ValueError, "with a 0-d mask"):
                    torchtrt.dynamo.compile(
                        torch.export.export(PutRow(), inputs),
                        arg_inputs=inputs,
                        min_block_size=1,
                        pass_through_build_failures=True,
                    )

        # int64 data stays exact when the mask is False.
        big = torch.tensor([2**40, 2**40 + 3], device="cuda")
        with self.subTest("int64_False"):

            class PutFive(torch.nn.Module):
                def forward(self, data, mask):
                    return torch.ops.aten.index_put.default(
                        data, [mask], torch.tensor(5)
                    )

            inputs = (big, torch.tensor(False, device="cuda"))
            compiled = torchtrt.dynamo.compile(
                torch.export.export(PutFive(), inputs),
                arg_inputs=inputs,
                min_block_size=1,
                pass_through_build_failures=True,
            )
            torch.testing.assert_close(compiled(*inputs), big)

        # Values with the input's dynamic batch dim fit the input.
        class PutScaled(torch.nn.Module):
            def forward(self, data, index):
                return torch.ops.aten.index_put.default(data, [index > 0], data * 10)

        batch = torch.export.Dim("batch", min=1, max=8)
        with self.subTest("dynamic_1"):
            compiled = torchtrt.dynamo.compile(
                torch.export.export(
                    PutScaled(), (data, positive), dynamic_shapes=({0: batch}, None)
                ),
                arg_inputs=[
                    torchtrt.Input(
                        min_shape=(1, 3),
                        opt_shape=(2, 3),
                        max_shape=(8, 3),
                        dtype=torch.float32,
                    ),
                    positive,
                ],
                min_block_size=1,
                pass_through_build_failures=True,
            )
            for rows in (2, 5):
                x = torch.arange(3.0 * rows, device="cuda").reshape(rows, 3)
                torch.testing.assert_close(compiled(x, positive), x * 10)

        # PyTorch ignores a 0-d mask next to another index, so that index_put runs there.
        class PutAtRows(torch.nn.Module):
            def forward(self, data, index, rows):
                return torch.ops.aten.index_put.default(
                    data, [index > 0, rows], torch.tensor(7.0)
                )

        rows = torch.tensor([1], device="cuda")
        with self.subTest("PutAtRows_1"):
            inputs = (data, positive, rows)
            compiled = torchtrt.dynamo.compile(
                torch.export.export(PutAtRows(), inputs),
                arg_inputs=inputs,
                min_block_size=1,
                pass_through_build_failures=True,
            )
            torch.testing.assert_close(compiled(*inputs), PutAtRows()(*inputs))

    def test_shapes_that_do_not_broadcast_raise(self):
        # torch.export accepts these, so the converter has to reject them as PyTorch does.
        class Put(torch.nn.Module):
            def forward(self, data, a, b, values):
                return torch.ops.aten.index_put.default(data, [a, b], values)

        class PutOne(torch.nn.Module):
            def forward(self, data, a, values):
                return torch.ops.aten.index_put.default(data, [a], values)

        class PutRows(torch.nn.Module):
            def forward(self, data, rows, columns, values):
                rows = rows.nonzero().squeeze(1)
                return torch.ops.aten.index_put.default(data, [rows, columns], values)

        class PutPairs(torch.nn.Module):
            def forward(self, data, rows, columns, values):
                rows, columns = rows.nonzero().squeeze(1), columns.nonzero().squeeze(1)
                return torch.ops.aten.index_put.default(data, [rows, columns], values)

        class PutMesh(torch.nn.Module):
            def forward(self, data, rows, columns, values):
                rows, columns = rows.nonzero(), columns.nonzero().T
                return torch.ops.aten.index_put.default(data, [rows, columns], values)

        def compile_(model, inputs):
            return torchtrt.dynamo.compile(
                torch.export.export(model, inputs),
                arg_inputs=inputs,
                min_block_size=1,
                pass_through_build_failures=True,
            )

        def cuda(*values):
            return torch.tensor(values, device="cuda")

        seven = torch.tensor(7.0, device="cuda")
        at_build = {
            "index shapes (1, 2) and (1, 3)": (
                Put(),
                (
                    torch.zeros(4, 4, device="cuda"),
                    cuda([0, 2]),
                    cuda([0, 1, 2]),
                    seven,
                ),
            ),
            "two values for rows of three": (
                PutOne(),
                (torch.zeros(4, 3, device="cuda"), cuda(0, 2), cuda(5.0, 7.0)),
            ),
            "no values for rows of three": (
                PutPairs(),
                (
                    torch.zeros(4, 4, 3, device="cuda"),
                    cuda(0, 0, 0, 0),
                    cuda(0, 0, 0, 0),
                    torch.zeros(0, device="cuda"),
                ),
            ),
            "values with a dim more than x[i]": (
                PutOne(),
                (
                    torch.zeros(4, device="cuda"),
                    torch.tensor(2, device="cuda"),
                    seven.reshape(1, 1),
                ),
            ),
            "a mask of two for an input of four": (
                PutOne(),
                (
                    torch.zeros(4, device="cuda"),
                    torch.tensor([True, True], device="cuda"),
                    seven,
                ),
            ),
            "two indices for an input of one dim": (
                Put(),
                (torch.zeros(4, device="cuda"), cuda(0), cuda(1), seven),
            ),
            "values of four for rows of three and an empty index": (
                PutOne(),
                (
                    torch.zeros(4, 3, device="cuda"),
                    torch.zeros(0, dtype=torch.int64, device="cuda"),
                    torch.ones(1, 4, device="cuda"),
                ),
            ),
        }
        for case, (model, inputs) in at_build.items():
            with self.subTest(case):
                if ENABLED_FEATURES.tensorrt_rtx and any(
                    t.dtype == torch.bool for t in inputs
                ):
                    self.skipTest(
                        "Skipped on tensorrt_rtx due to nonzero not supported"
                    )
                with self.assertRaises((ValueError, RuntimeError)):
                    compile_(model, inputs)

        # Sizes that disagree only at runtime fail there.
        data = torch.zeros(4, 4, device="cuda")
        two, three = cuda(1, 0, 1, 0), cuda(1, 1, 1, 0)
        ones = torch.ones(2, 3, device="cuda")
        at_runtime = {
            "three rows and two fixed columns": (
                PutRows(),
                (two, cuda(0, 2), seven),
                (three, cuda(0, 2), seven),
            ),
            "two rows and three selected columns": (
                PutPairs(),
                (two, cuda(0, 1, 0, 1), seven),
                (two, three, seven),
            ),
            "values of shape (2, 3) for three rows and two columns": (
                PutMesh(),
                (two, three, ones),
                (three, two, ones),
            ),
        }
        for case, (model, valid, invalid) in at_runtime.items():
            compiled = compile_(model, (data, *valid))
            with self.subTest(case):
                torch.testing.assert_close(compiled(data, *valid), model(data, *valid))
                with self.assertRaises((RuntimeError, IndexError)):
                    compiled(data, *invalid)

    def test_one_element_cpu_value_fills_a_mask_whatever_its_shape(self):
        # PyTorch runs this as masked_fill, which takes a one-element CPU value as a scalar.
        class Fill(torch.nn.Module):
            def forward(self, data, mask):
                value = torch.full((1, 1, 1), 7.0)
                return torch.ops.aten.index_put.default(data, [None, mask], value)

        # torch.export takes the CPU value only next to CPU inputs.
        exported = torch.export.export(
            Fill(), (torch.zeros(2, 1, 1), torch.ones(1, 1, dtype=torch.bool))
        )
        data = torch.zeros(2, 1, 1, device="cuda")
        masks = [
            torch.tensor([[True]], device="cuda"),
            torch.tensor([[False]], device="cuda"),
        ]
        compiled = torchtrt.dynamo.compile(
            exported,
            arg_inputs=(data, masks[0]),
            min_block_size=1,
            pass_through_build_failures=True,
        )
        for mask in masks:
            torch.testing.assert_close(compiled(data, mask), Fill()(data, mask))

        # masked_fill also ignores a None after the mask.
        class FillWithTrailingNone(torch.nn.Module):
            def forward(self, data, mask):
                value = torch.tensor(7.0)
                return torch.ops.aten.index_put.default(data, [mask, None], value)

        inputs = (
            torch.zeros(2, 3, device="cuda"),
            torch.tensor([[True, False, True], [False, True, False]], device="cuda"),
        )
        compiled = torchtrt.dynamo.compile(
            torch.export.export(FillWithTrailingNone(), inputs),
            arg_inputs=inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        torch.testing.assert_close(compiled(*inputs), FillWithTrailingNone()(*inputs))

    def test_updates_are_cast_to_int64_destination_dtype(self):
        class IndexPutArgmax(torch.nn.Module):
            def forward(self, data, idx, scores):
                values = torch.ops.aten.argmax.default(scores, 1)
                return torch.ops.aten.index_put.default(data, [idx], values)

        data = torch.zeros(8, dtype=torch.int64, device="cuda")
        idx = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        scores = torch.randn((2, 5), device="cuda")
        model = IndexPutArgmax().eval().cuda()
        expected = model(data, idx, scores)

        exported = torch.export.export(model, (data, idx, scores))
        compiled = torchtrt.dynamo.compile(
            exported,
            arg_inputs=[data, idx, scores],
            min_block_size=1,
            pass_through_build_failures=True,
        )

        self.assertEqual(compiled(data, idx, scores), expected)

    def test_dynamic_index_broadcasts_1d_values_across_free_dim(self):
        class IndexPutFreeDim(torch.nn.Module):
            def forward(self, data, idx):
                values = idx.to(torch.float32)
                return torch.ops.aten.index_put.default(data, [None, idx], values)

        data = torch.zeros((32, 64), device="cuda")
        idx = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device="cuda")
        index_length = torch.export.Dim("index_length", min=1, max=16)
        model = IndexPutFreeDim().eval().cuda()

        exported = torch.export.export(
            model,
            (data, idx),
            dynamic_shapes={"data": {}, "idx": {0: index_length}},
        )
        compiled = torchtrt.dynamo.compile(
            exported,
            arg_inputs=[
                torchtrt.Input(shape=(32, 64), dtype=torch.float32),
                torchtrt.Input(
                    min_shape=(1,),
                    opt_shape=(4,),
                    max_shape=(16,),
                    dtype=torch.int64,
                ),
            ],
            min_block_size=1,
            pass_through_build_failures=True,
        )

        for runtime_idx in (
            idx,
            torch.tensor([2, 6], dtype=torch.int64, device="cuda"),
        ):
            self.assertEqual(
                compiled(data, runtime_idx),
                model(data, runtime_idx),
            )
        torch._dynamo.reset()
        compile_idx = idx.clone()
        torch._dynamo.mark_dynamic(compile_idx, 0)
        compiled_backend = torch.compile(
            model,
            backend="tensorrt",
            options={"pass_through_build_failures": True, "min_block_size": 1},
        )
        self.assertEqual(
            compiled_backend(data, compile_idx),
            model(data, compile_idx),
        )

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

    @parameterized.expand([(torch.float32,), (torch.float16,), (torch.bfloat16,)])
    def test_accumulate_needs_no_pytorch_plugin(self, dtype):
        """With fallback_pytorch_plugin_ops, TensorRT sums duplicates and broadcasts values as eager."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return x.index_put((index,), values, accumulate=True)

        # All 16 indices are row 1, and one 16 x 2 block of values broadcasts over them. Rounding to
        # 16 bits after each add would keep the 1.0 entries at 2048, so only a float32 sum gives 2064.
        inputs = (
            torch.full((3, 16, 2), 2048.0, dtype=dtype, device="cuda"),
            torch.tensor([1] * 15 + [-2], device="cuda"),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]] * 8, dtype=dtype, device="cuda"),
        )

        def compile_with(**settings):
            compiled = torchtrt.dynamo.compile(
                torch.export.export(Accumulate(), inputs),
                arg_inputs=inputs,
                min_block_size=1,
                **settings,
            )
            engines = [
                module
                for module in compiled.modules()
                if isinstance(module, torchtrt.dynamo.runtime.TorchTensorRTModule)
            ]
            return compiled, engines

        # Without the setting nothing changes: ScatterAdd, or PyTorch where it is missing.
        _, engines = compile_with()
        self.assertTrue(all(engine.pytorch_plugin_ops for engine in engines))
        trt_mod, (engine,) = compile_with(fallback_pytorch_plugin_ops=True)
        self.assertEqual(engine.pytorch_plugin_ops, [])
        # Version 1 is an IPluginV2 and stays its own layer. TensorRT on Thor merges the
        # IPluginV3 version 2 into a compiler region that needs far more build memory.
        import tensorrt

        v1 = tensorrt.get_plugin_registry().get_creator("ScatterElements", "1", "")
        self.assertIn(
            "[PLUGIN_V2]" if v1 is not None else "[PLUGIN_V3]", engine.get_layer_info()
        )
        torch.testing.assert_close(
            trt_mod(*inputs), Accumulate()(*inputs), rtol=0, atol=0
        )

    def test_accumulate_values_that_do_not_broadcast_raise(self):
        """With fallback_pytorch_plugin_ops, values that do not broadcast to x[index] raise, as in eager."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return x.index_put((index,), values, accumulate=True)

        # Two values for rows of three.
        inputs = (
            torch.zeros(4, 3, device="cuda"),
            torch.tensor([0, 2], device="cuda"),
            torch.tensor([5.0, 7.0], device="cuda"),
        )
        with self.assertRaises(RuntimeError):
            Accumulate()(*inputs)
        with self.assertRaisesRegex(ValueError, "do not broadcast"):
            torchtrt.dynamo.compile(
                torch.export.export(Accumulate(), inputs),
                arg_inputs=inputs,
                min_block_size=1,
                fallback_pytorch_plugin_ops=True,
                pass_through_build_failures=True,
            )

        # Four values for rows of three, a count known only at runtime, fail there.
        class AccumulateSelected(torch.nn.Module):
            def forward(self, x, index, values, mask):
                return x.index_put(
                    (index,), values[mask.nonzero().squeeze(1)], accumulate=True
                )

        x, index = inputs[:2]
        values = torch.tensor([5.0, 7.0, 9.0, 11.0], device="cuda")
        three, four = (
            torch.tensor([True, True, True, False], device="cuda"),
            torch.tensor([True, True, True, True], device="cuda"),
        )
        compiled = torchtrt.dynamo.compile(
            torch.export.export(AccumulateSelected(), (x, index, values, three)),
            arg_inputs=(x, index, values, three),
            min_block_size=1,
            fallback_pytorch_plugin_ops=True,
            pass_through_build_failures=True,
        )
        torch.testing.assert_close(
            compiled(x, index, values, three),
            AccumulateSelected()(x, index, values, three),
        )
        with self.assertRaises((RuntimeError, IndexError)):
            compiled(x, index, values, four)

    def test_accumulate_index_forms_with_fallback_setting(self):
        """With fallback_pytorch_plugin_ops, a bool mask stays in TensorRT and a 2-D index does not."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return x.index_put((index,), values, accumulate=True)

        cases = {
            "bool mask": (
                torch.zeros(5, 3, device="cuda"),
                torch.tensor([True, False, True, True, False], device="cuda"),
                torch.arange(9.0, device="cuda").reshape(3, 3),
                0,
            ),
            # A 2-D index, as in an embedding gradient, runs in PyTorch.
            "2-D index": (
                torch.zeros(10, 4, device="cuda"),
                torch.tensor([[1, 3, 3], [0, 9, 1]], device="cuda"),
                torch.ones(2, 3, 4, device="cuda"),
                1,
            ),
        }
        for name, (x, index, values, left_to_pytorch) in cases.items():
            with self.subTest(name):
                inputs = (x, index, values)
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(Accumulate(), inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    fallback_pytorch_plugin_ops=True,
                )
                left = [
                    node
                    for module in compiled.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                    if node.target is torch.ops.aten.index_put.default
                ]
                self.assertEqual(len(left), left_to_pytorch)
                torch.testing.assert_close(compiled(*inputs), Accumulate()(*inputs))

    def test_accumulate_scatter_add_leaves_mask_and_leading_none_to_pytorch(self):
        """ScatterAdd drops None and casts a bool mask to indices, so these run in PyTorch."""
        import tensorrt

        if (
            tensorrt.get_plugin_registry().get_creator(
                "ScatterAdd", "1", "torch_tensorrt"
            )
            is None
        ):
            pytest.skip("needs Torch-TensorRT's ScatterAdd plugin")

        class ColumnAccumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return torch.ops.aten.index_put.default(x, [None, index], values, True)

        class MaskAccumulate(torch.nn.Module):
            def forward(self, x, mask, values):
                return x.index_put((mask,), values, accumulate=True)

        cases = {
            "None before the index": (
                ColumnAccumulate(),
                torch.zeros(4, 8, device="cuda"),
                torch.tensor([1, 1, 5], device="cuda"),
                torch.ones(4, 3, device="cuda"),
            ),
            "bool mask": (
                MaskAccumulate(),
                torch.zeros(6, 4, device="cuda"),
                torch.tensor([True, False, True, True, False, True], device="cuda"),
                torch.ones(4, 4, device="cuda"),
            ),
        }
        for name, (model, *inputs) in cases.items():
            with self.subTest(name):
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(model, tuple(inputs)),
                    arg_inputs=inputs,
                    min_block_size=1,
                )
                # Checked before running, since ScatterAdd aborts the process on these.
                left = [
                    node
                    for module in compiled.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                    if node.target is torch.ops.aten.index_put.default
                ]
                self.assertEqual(len(left), 1)
                torch.testing.assert_close(compiled(*inputs), model(*inputs))

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

    def test_zero_size_free_dimension_no_op(self):
        """A size of 0 in a dimension that is not indexed also leaves nothing to write."""

        class PutRows(torch.nn.Module):
            def forward(self, src, values, idx):
                return torch.ops.aten.index_put.default(src, [idx], values)

        inputs = (
            torch.zeros(4, 0, device="cuda"),
            torch.zeros(2, 0, device="cuda"),
            torch.tensor([1, 2], device="cuda"),
        )
        trt_mod = torchtrt.dynamo.compile(
            torch.export.export(PutRows(), inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
        )
        torch.testing.assert_close(trt_mod(*inputs), PutRows()(*inputs))

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

    def test_accumulate_dynamic_shape_that_can_reach_2_31_elements(self):
        """With fallback_pytorch_plugin_ops, a dynamic shape that can reach 2**31 elements runs in PyTorch."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return x.index_put((index,), values, accumulate=True)

        # ScatterElements reads past its buffers once the updates reach 2**31 elements. The
        # validator sees only the exported range, so an unbounded one cannot be trusted. A long
        # index makes the updates larger than the input, and more rows make the input larger.
        for rows, max_columns, exported_max, index_length, left_to_pytorch in (
            (4, 2**20, 2**20, 4, 0),
            (4, 2**29, 2**29, 4, 1),
            (4, 2**20, None, 4, 1),
            (4, 2**20, 2**20, 4096, 1),
            (8, 2**28, 2**28, 4, 1),
        ):
            inputs = (
                torch.zeros(rows, 64, device="cuda"),
                torch.tensor([0, 0, 1, 3], device="cuda").repeat(index_length // 4),
                torch.ones(1, 1, device="cuda"),
            )
            with self.subTest(
                rows=rows,
                max_columns=max_columns,
                exported_max=exported_max,
                index_length=index_length,
            ):
                columns = torch.export.Dim("columns", min=2, max=exported_max)
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(
                        Accumulate(),
                        inputs,
                        dynamic_shapes=({1: columns}, None, None),
                    ),
                    arg_inputs=[
                        torchtrt.Input(
                            min_shape=(rows, 2),
                            opt_shape=(rows, 64),
                            max_shape=(rows, max_columns),
                            dtype=torch.float32,
                        ),
                        *inputs[1:],
                    ],
                    min_block_size=1,
                    fallback_pytorch_plugin_ops=True,
                )
                left = [
                    node
                    for module in compiled.modules()
                    if isinstance(module, torch.fx.GraphModule)
                    for node in module.graph.nodes
                    if node.target is torch.ops.aten.index_put.default
                ]
                self.assertEqual(len(left), left_to_pytorch)
                torch.testing.assert_close(compiled(*inputs), Accumulate()(*inputs))

    def test_accumulate_mask_of_another_length_with_fallback_setting(self):
        """A bool mask that does not match its dimension stays in PyTorch, which rejects it."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, mask, values):
                return x.index_put((mask,), values, accumulate=True)

        for length in (3, 5):
            with self.subTest(length=length):
                mask = torch.tensor([True, False] * 3, device="cuda")[:length]
                inputs = (
                    torch.zeros(4, device="cuda"),
                    mask,
                    torch.tensor(7.0, device="cuda"),
                )
                compiled = torchtrt.dynamo.compile(
                    torch.export.export(Accumulate(), inputs),
                    arg_inputs=inputs,
                    min_block_size=1,
                    fallback_pytorch_plugin_ops=True,
                )
                with self.assertRaises(IndexError):
                    Accumulate()(*inputs)
                with self.assertRaises(IndexError):
                    compiled(*inputs)

    def test_accumulate_serialized_engine_needs_no_pytorch_plugin(self):
        """The serialized engine export honors fallback_pytorch_plugin_ops too."""
        if ENABLED_FEATURES.tensorrt_rtx:
            pytest.skip("TensorRT's ScatterElements plugin is not available in TRT RTX")

        class Accumulate(torch.nn.Module):
            def forward(self, x, index, values):
                return x.index_put((index,), values, accumulate=True)

        inputs = (
            torch.zeros(4, 3, device="cuda"),
            torch.tensor([0, 2, 2], device="cuda"),
            torch.ones(3, 3, device="cuda"),
        )
        engine = torchtrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            torch.export.export(Accumulate(), inputs),
            arg_inputs=inputs,
            min_block_size=1,
            fallback_pytorch_plugin_ops=True,
        )
        self.assertFalse(b"ScatterAdd" in engine)
        self.assertTrue(b"ScatterElements" in engine)
        trt_mod = torchtrt.dynamo.runtime.TorchTensorRTModule(
            engine, ["x", "index", "values"], ["output0"]
        )
        torch.testing.assert_close(trt_mod(*inputs), Accumulate()(*inputs))

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
        )


if __name__ == "__main__":
    run_tests()
