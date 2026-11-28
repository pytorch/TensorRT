# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep statically known slice extents through dynamic TensorRT conversion."""

import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("prefix_value", [0.0, 1.0])
def test_dynamic_table_slice_before_attention(prefix_value):
    # Issue #5037: the table's length is dynamic, but table[:, :1] is
    # always [1, 1, 1]. Losing that singleton breaks K/V shape equality.
    # Also exercise a nonzero scale so attention numerics are not trivial.
    class Model(torch.nn.Module):
        def forward(self, x, ids):
            table = torch.cat(
                [
                    torch.full((4,), prefix_value, device=x.device),
                    ids.float() + 7,
                    torch.ones(8, device=x.device),
                ]
            )[None, :, None]
            table = table[:, : x.shape[1]]
            for _ in range(2):
                q = (x * table).transpose(0, 1)[None]
                k = (x * table).transpose(0, 1)[None]
                v = x.transpose(0, 1)[None]
                x = x + F.scaled_dot_product_attention(q, k, v)[0].transpose(0, 1)
            return x

    def inputs(n):
        return (
            torch.randn(n + 12, 1, 16, device="cuda") * 0.1,
            torch.arange(1, n + 1, device="cuda"),
        )

    model = Model().eval().cuda()
    example = inputs(20)
    n = torch.export.Dim("n", min=2, max=28)
    exported = torch.export.export(
        model, example, dynamic_shapes=({0: n + 12}, {0: n}), strict=False
    )
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=example,
        min_block_size=1,
        require_full_compilation=True,
        disable_tf32=True,
    )
    for length in (2, 5, 20, 28):
        args = inputs(length)
        torch.testing.assert_close(compiled(*args), model(*args), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_fixed_length_narrow_at_symbolic_offset_then_split(dynamic_batch):
    # The start moves with the input, but the split axis is always 12.
    # A separate dynamic batch must not be specialized to the export example.
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.narrow(1, x.shape[1] - 12, 12).split([8, 4], dim=1)

    model = Model().eval().cuda()
    example = (torch.randn(2 if dynamic_batch else 1, 32, 4, device="cuda"),)
    dynamic_dims = {1: torch.export.Dim("length", min=14, max=40)}
    if dynamic_batch:
        dynamic_dims[0] = torch.export.Dim("batch", min=2, max=4)
    exported = torch.export.export(
        model, example, dynamic_shapes=(dynamic_dims,), strict=False
    )
    compiled = torch_tensorrt.dynamo.compile(
        exported, arg_inputs=example, min_block_size=1, require_full_compilation=True
    )
    for batch, length in ((2, 14), (3, 17), (2, 32), (4, 40)):
        x = torch.randn(batch if dynamic_batch else 1, length, 4, device="cuda")
        actual = compiled(x)
        torch.testing.assert_close(actual, model(x))
        assert actual[0].shape == (x.shape[0], 8, 4)
        assert actual[1].shape == (x.shape[0], 4, 4)


def test_slice_with_symbolic_output_extent_is_not_specialized():
    # The export example produces 30 elements, but smaller inputs produce
    # fewer. Do not freeze a symbolic size to its example.
    class Model(torch.nn.Module):
        def forward(self, x):
            return x[:, 2:]

    model = Model().eval().cuda()
    example = (torch.randn(2, 32, device="cuda"),)
    exported = torch.export.export(
        model,
        example,
        dynamic_shapes=({1: torch.export.Dim("length", min=4, max=64)},),
        strict=False,
    )
    compiled = torch_tensorrt.dynamo.compile(
        exported, arg_inputs=example, min_block_size=1, require_full_compilation=True
    )
    for length in (4, 5, 8, 32, 64):
        x = torch.randn(2, length, device="cuda")
        torch.testing.assert_close(compiled(x), model(x))
