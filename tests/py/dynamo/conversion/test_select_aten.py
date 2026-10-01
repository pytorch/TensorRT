# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSelectConverterOne(DispatchTestCase):
    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_2d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(1, 2)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            ("dim_index", 1, 0),
        ]
    )
    def test_select_4d(self, _, dim, index):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(4, 4, 4, 4)]
        self.run_test(
            select(),
            input,
        )

    @parameterized.expand(
        [
            (
                "partial_dynamic_static_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                2,
                0,
            ),
            (
                "partial_dynamic_dynamic_dim",
                (1, 1, 3),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                1,
                1,
            ),
            (
                "fully_dynamic_neg_dim",
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                -1,
                1,
            ),
        ]
    )
    def test_dynamic_shape_select(
        self, _, min_shape, opt_shape, max_shape, type, dim, index
    ):
        class select(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(select(), input_specs)

    @parameterized.expand(
        [
            ("rank0_index", [1, 2], [0], False),
            ("rank1_size1_index", [1, 2], [0], True),
            ("rank2_size1_index", [[1, 2]], [0, 1], True),
        ]
    )
    def test_select_runtime_index(self, _, positions, dims, keepdim):
        """Select row 3 of an (8, 4) input with an index computed in the engine.

        Eager gives shape (4,) for any index with one element. A rank 0 index is only
        cast. A (1,) index, the shape an integer scalar has after crossing a partition
        boundary, and a (1, 1) index from a keepdim reduction are reshaped to rank 0.
        """

        class SelectRuntimeIndex(nn.Module):
            def forward(self, values, positions):
                index = torch.ops.aten.sum.dim_IntList(positions, dims, keepdim)
                return torch.ops.aten.select.int(values, 0, index)

        inputs = [
            torch.randn(8, 4),
            torch.tensor(positions, dtype=torch.int64),
        ]
        # torch.export rejects a tensor index for select, so trace with symbolic_trace.
        self.run_test(SelectRuntimeIndex(), inputs, use_dynamo_tracer=False)

    @parameterized.expand(
        [
            ("two_elements", Input(shape=(2,), dtype=torch.int64)),
            (
                "dynamic_size",
                Input(
                    min_shape=(1,), opt_shape=(1,), max_shape=(2,), dtype=torch.int64
                ),
            ),
        ]
    )
    def test_select_runtime_index_not_single_element(self, _, index_spec):
        class SelectRuntimeIndex(nn.Module):
            def forward(self, values, index):
                return torch.ops.aten.select.int(values, 0, index)

        input_specs = [Input(shape=(8, 4), dtype=torch.float32), index_spec]
        # Eager rejects a two-element index as well, so skip the dtype check that runs it.
        with self.assertRaisesRegex(RuntimeError, "exactly one element"):
            self.run_test_with_dynamic_shape(
                SelectRuntimeIndex(),
                input_specs,
                use_dynamo_tracer=False,
                check_dtype=False,
            )


if __name__ == "__main__":
    run_tests()
