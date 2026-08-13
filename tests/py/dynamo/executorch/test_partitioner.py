from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch.exir")

from torch_tensorrt.executorch.partitioner import TensorRTPartitioner  # noqa: E402


@pytest.mark.unit
def test_partitioner_tags_proposed_partitions(monkeypatch):
    class FakeCapabilityPartitioner:
        def __init__(
            self, graph_module, operator_support, allows_single_node_partition
        ):
            self.graph_module = graph_module
            self.operator_support = operator_support
            self.allows_single_node_partition = allows_single_node_partition

        def propose_partitions(self):
            node_a = SimpleNamespace(meta={})
            node_b = SimpleNamespace(meta={})
            return [
                SimpleNamespace(id=1, nodes=[node_a]),
                SimpleNamespace(id=2, nodes=[node_b]),
            ]

    tagged = {"called": False}

    def fake_tag_constant_data(exported_program):
        tagged["called"] = True

    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner",
        FakeCapabilityPartitioner,
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.tag_constant_data",
        fake_tag_constant_data,
    )

    graph_module = SimpleNamespace(graph=SimpleNamespace(nodes=[]))
    exported_program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
    )

    result = TensorRTPartitioner().partition(exported_program)

    assert tagged["called"]
    assert sorted(result.partition_tags.keys()) == ["tensorrt_1", "tensorrt_2"]


@pytest.mark.unit
def test_keep_mutated_buffers_above_delegate_untags_only_mutation_targets():
    """The un-tag post-pass keeps a delegate-mutated buffer above the delegate
    (strips its delegation_tag) while leaving non-mutated constants tagged into
    the delegate and non-placeholder nodes untouched.
    """
    from torch_tensorrt.executorch.partitioner import (
        _keep_mutated_buffers_above_delegate,
    )

    mutated_buf = SimpleNamespace(
        op="placeholder", name="b_k_0", meta={"delegation_tag": "tensorrt_0"}
    )
    const_buf = SimpleNamespace(
        op="placeholder", name="b_w", meta={"delegation_tag": "tensorrt_0"}
    )
    engine_node = SimpleNamespace(
        op="call_function", name="tensorrt_0", meta={"delegation_tag": "tensorrt_0"}
    )
    exported_program = SimpleNamespace(
        graph_module=SimpleNamespace(
            graph=SimpleNamespace(nodes=[mutated_buf, const_buf, engine_node])
        ),
        graph_signature=SimpleNamespace(
            buffers_to_mutate={"getitem_5": "k_0"},
            inputs_to_buffers={"b_k_0": "k_0", "b_w": "w"},
        ),
    )

    _keep_mutated_buffers_above_delegate(exported_program)

    # k_0 is a mutation target -> its buffer placeholder is kept above the delegate
    assert "delegation_tag" not in mutated_buf.meta
    # w is not mutated -> still frozen into the delegate
    assert const_buf.meta["delegation_tag"] == "tensorrt_0"
    # non-placeholder nodes are untouched
    assert engine_node.meta["delegation_tag"] == "tensorrt_0"
