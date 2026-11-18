# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils._ordered_set import OrderedSet
from torch_tensorrt.dynamo.lowering.passes._FakeTensorUpdater import (
    FakeTensorUpdater,
    _torch_FakeTensorUpdater,
)


def _graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    relu = graph.call_function(torch.ops.aten.relu.default, (neg,))
    graph.output(relu)
    return torch.fx.GraphModule({}, graph), neg, relu


@pytest.mark.parametrize("storage_type", [dict, OrderedSet])
def test_only_nodes_missing_values_are_invalidated(monkeypatch, storage_type):
    gm, missing, populated = _graph()
    populated.meta["val"] = torch.ones(2, 3)
    original_init = _torch_FakeTensorUpdater.__init__

    def init_with_storage(self, gm):
        original_init(self, gm)
        # Exercise both upstream representations regardless of installed torch.
        hashes = {node: self.hash_node(node) for node in gm.graph.nodes}
        self.processed_hashes = (
            hashes if storage_type is dict else OrderedSet(hashes.values())
        )

    monkeypatch.setattr(_torch_FakeTensorUpdater, "__init__", init_with_storage)
    updater = FakeTensorUpdater(gm)
    retained = [node for node in gm.graph.nodes if node is not missing]
    if storage_type is dict:
        assert updater.processed_hashes == {
            node: updater.hash_node(node) for node in retained
        }
    else:
        assert updater.processed_hashes == OrderedSet(
            updater.hash_node(node) for node in retained
        )


def test_incremental_update_restores_missing_fake_value():
    gm, missing, populated = _graph()
    fake_mode = FakeTensorMode()
    FakeTensorProp(gm, fake_mode).propagate(torch.randn(2, 3))
    expected = missing.meta.pop("val")

    updater = FakeTensorUpdater(gm)
    updater.incremental_update(fake_mode)

    restored = missing.meta["val"]
    assert isinstance(restored, FakeTensor)
    assert restored.fake_mode is fake_mode
    assert restored.shape == expected.shape
    assert restored.stride() == expected.stride()
    assert restored.dtype == expected.dtype
    assert populated.meta["val"].shape == restored.shape
