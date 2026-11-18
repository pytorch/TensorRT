# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for metadata refresh, independent of native TRT extensions."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def updater_module():
    path = ROOT / "py/torch_tensorrt/dynamo/lowering/passes/_FakeTensorUpdater.py"
    spec = importlib.util.spec_from_file_location("trt_fake_tensor_updater_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exported_graph():
    class Model(torch.nn.Module):
        def forward(self, x):
            return (x + 1).relu()

    return torch.export.export(Model(), (torch.randn(2, 3),)).module()


def test_missing_fake_metadata_is_recomputed(updater_module):
    graph = exported_graph()
    node = next(n for n in graph.graph.nodes if n.op == "call_function")
    expected = node.meta.pop("val")
    updater = updater_module.FakeTensorUpdater(graph)
    updater.incremental_update(expected.fake_mode)
    actual = node.meta["val"]
    assert isinstance(actual, type(expected))
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert actual.fake_mode is expected.fake_mode


def test_known_metadata_remains_cached(updater_module):
    graph = exported_graph()
    known = {n: n.meta["val"] for n in graph.graph.nodes if "val" in n.meta}
    updater = updater_module.FakeTensorUpdater(graph)
    fake_mode = next(v.fake_mode for v in known.values() if isinstance(v, torch.Tensor))
    updater.incremental_update(fake_mode)
    for node, expected in known.items():
        assert node.meta["val"] is expected


@pytest.mark.parametrize("precache_missing", [False, True])
def test_node_keyed_cache_invalidates_only_missing_metadata(
    updater_module, monkeypatch, precache_missing
):
    graph = exported_graph()
    missing = next(n for n in graph.graph.nodes if n.op == "call_function")
    del missing.meta["val"]
    parent = updater_module._torch_FakeTensorUpdater

    def node_keyed_init(self, gm):
        # Exercise the newer constructor/cache contract on older torch as well.
        # Real incremental propagation is covered separately on the runtime's
        # actual parent implementation above.
        self.gm = gm
        self.processed_hashes = {
            node: self.hash_node(node)
            for node in gm.graph.nodes
            if node is not missing or precache_missing
        }

    monkeypatch.setattr(parent, "__init__", node_keyed_init)
    updater = updater_module.FakeTensorUpdater(graph)
    assert missing not in updater.processed_hashes
    assert set(updater.processed_hashes) == set(graph.graph.nodes) - {missing}
    for node, value in updater.processed_hashes.items():
        assert value == updater.hash_node(node)
