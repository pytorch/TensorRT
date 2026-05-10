from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch")

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
    exported_program = SimpleNamespace(graph_module=graph_module)

    result = TensorRTPartitioner().partition(exported_program)

    assert tagged["called"]
    assert sorted(result.partition_tags.keys()) == ["tensorrt_1", "tensorrt_2"]
