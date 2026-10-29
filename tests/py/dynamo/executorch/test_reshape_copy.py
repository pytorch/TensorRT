# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch_tensorrt.executorch._export import _normalize_reshape_copy

pytestmark = pytest.mark.unit


class CopyReshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.ops.aten._reshape_copy.default(x, self.shape)


@pytest.mark.parametrize(
    "x,shape",
    [
        (torch.arange(24.0).reshape(2, 3, 4), [3, 8]),
        (torch.arange(24.0).reshape(2, 3, 4).transpose(0, 1), [3, 2, 4]),
        (torch.arange(24.0).reshape(2, 3, 4).transpose(0, 1), [3, 8]),
        (torch.ones(1, 4).expand(3, 4), [2, 6]),
        (torch.empty(0, 4), [0, 2, 2]),
        (torch.tensor(3.0), []),
    ],
)
def test_copy_reshape_preserves_values_layout_and_non_aliasing(x, shape):
    source = torch.export.export(CopyReshape(shape), (x,))
    source_graph = str(source.graph)
    normalized = _normalize_reshape_copy(source)
    normalized.validate()
    assert str(source.graph) == source_graph
    assert all(
        node.target != torch.ops.aten._reshape_copy.default
        for node in normalized.graph.nodes
    )
    actual = normalized.module()(x)
    expected = source.module()(x)
    torch.testing.assert_close(actual, expected)
    assert actual.stride() == expected.stride()
    assert actual.is_contiguous()
    if x.numel():
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        original = x.clone()
        actual.add_(1)
        torch.testing.assert_close(x, original)


def test_copy_reshape_preserves_dynamic_shapes_and_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            return torch.ops.aten._reshape_copy.default(x + self.weight, [-1, 2])

    source = torch.export.export(
        Model(),
        (torch.randn(3, 4),),
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2, max=10)}},
    )
    normalized = _normalize_reshape_copy(source)
    normalized.validate()
    assert normalized.state_dict["weight"] is source.state_dict["weight"]
    assert normalized.range_constraints == source.range_constraints
    for batch in (2, 3, 7):
        x = torch.randn(batch, 4)
        torch.testing.assert_close(normalized.module()(x), source.module()(x))


def test_copy_reshape_leaves_unaffected_program_untouched(monkeypatch):
    source = torch.export.export(torch.nn.ReLU(), (torch.randn(2, 3),))

    def unexpected_retrace(*args, **kwargs):
        raise AssertionError("programs without _reshape_copy must not be retraced")

    monkeypatch.setattr(source, "run_decompositions", unexpected_retrace)
    assert _normalize_reshape_copy(source) is source


def test_copy_reshape_preserves_mutation_signature():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(2, 4))

        def forward(self, x):
            copied = torch.ops.aten._reshape_copy.default(x, [2, 4])
            self.cache.copy_(copied)
            x.add_(1)
            return copied

    source = torch.export.export(Model(), (torch.randn(4, 2),)).run_decompositions({})
    normalized = _normalize_reshape_copy(source)
    normalized.validate()
    assert [spec.kind for spec in normalized.graph_signature.output_specs] == [
        spec.kind for spec in source.graph_signature.output_specs
    ]
    # The producing node is now named clone, not _reshape_copy. The mutation
    # target and output kind must remain the same, not the internal node name.
    assert [spec.target for spec in normalized.graph_signature.output_specs] == [
        spec.target for spec in source.graph_signature.output_specs
    ]
    x = torch.randn(4, 2)
    original = x.clone()
    actual = normalized.module()(x)
    torch.testing.assert_close(actual, original.reshape(2, 4))
    torch.testing.assert_close(x, original + 1)
    torch.testing.assert_close(normalized.state_dict["cache"], actual)
