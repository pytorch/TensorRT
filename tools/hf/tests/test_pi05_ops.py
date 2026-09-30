import pytest
import torch
from torch import nn
from torch_tensorrt_edge_llm import ops
from torch_tensorrt_edge_llm.serialization import EdgeComponentMetadata, EdgeOutputSpec


def metadata(component, runner, shapes):
    return EdgeComponentMetadata(
        component=component,
        runner=runner,
        outputs=tuple(EdgeOutputSpec(shape=shape, dtype="float32") for shape in shapes),
    ).to_json()


def test_prefill_and_repeated_action_execution_keep_prefix_read_only(monkeypatch):
    class Prefill(nn.Module):
        def forward(self, embeds, mask, positions):
            hidden = embeds + positions.unsqueeze(-1)
            return (
                hidden,
                hidden.unsqueeze(0).unsqueeze(2).clone(),
                (hidden * 2).unsqueeze(0).unsqueeze(2),
            )

    class Action(nn.Module):
        def forward(self, x, time, k, v, positions, mask):
            return x * time[:, None, None] + k.mean() + v.mean()

    monkeypatch.setattr(
        ops,
        "_get_embedded_engine",
        lambda blob, meta: Prefill() if meta.component == "language" else Action(),
    )
    embeds = torch.randn(1, 3, 2)
    positions = torch.arange(3).unsqueeze(0)
    blob = torch.zeros(1, dtype=torch.uint8)
    _, k, v = ops.llm_prefill(
        embeds,
        torch.zeros(1, 1, 3, 3),
        positions,
        blob,
        metadata(
            "language", "pi05_prefill", [(1, 3, 2), (1, 1, 1, 3, 2), (1, 1, 1, 3, 2)]
        ),
    )
    before = k.clone(), v.clone()
    action_meta = metadata("action", "pi05_action", [(1, 2, 2)])
    x = torch.randn(1, 2, 2)
    for step in range(3):
        time = torch.full((1,), 1 - step / 3)
        velocity = ops.action_expert(
            x,
            time,
            k,
            v,
            torch.arange(2).unsqueeze(0),
            torch.zeros(1, 1, 2, 5),
            blob,
            action_meta,
        )
        torch.testing.assert_close(velocity, Action()(x, time, k, v, None, None))
        x = x - velocity / 3
    torch.testing.assert_close(k, before[0])
    torch.testing.assert_close(v, before[1])


def test_prefill_rejects_other_runtime_contract():
    value = torch.ones(1, 3, 2)
    with pytest.raises(ValueError, match="pi05_prefill"):
        ops.llm_prefill(
            value,
            torch.zeros(1, 1, 3, 3),
            torch.arange(3).unsqueeze(0),
            torch.zeros(1, dtype=torch.uint8),
            metadata("language", "autoregressive", [(1, 3, 2)]),
        )


def test_pi05_components_lower_with_constants_consumed_by_delegate(tmp_path):
    pytest.importorskip("executorch.exir")
    from executorch.exir._serialize._program import deserialize_pte_binary
    from torch_tensorrt import save
    from torch_tensorrt.executorch.serialization import (
        TensorRTBlobMetadata,
        TensorRTIOBinding,
        serialize_engine,
    )
    from torch_tensorrt_edge_llm import EdgeLLMPartitioner
    from torch_tensorrt_edge_llm.serialization import deserialize_edge_component

    class PrefixAndAction(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefill_metadata = metadata(
                "language",
                "pi05_prefill",
                [(1, 3, 2), (1, 1, 1, 3, 2), (1, 1, 1, 3, 2)],
            )
            self.action_metadata = metadata("action", "pi05_action", [(1, 2, 2)])
            for name, inputs, outputs in [("prefill", 3, 3), ("action", 6, 1)]:
                payload = serialize_engine(
                    b"test-engine",
                    TensorRTBlobMetadata(
                        io_bindings=[
                            *[
                                TensorRTIOBinding(f"in{i}", is_input=True)
                                for i in range(inputs)
                            ],
                            *[
                                TensorRTIOBinding(f"out{i}", is_input=False)
                                for i in range(outputs)
                            ],
                        ]
                    ),
                )
                self.register_buffer(
                    name, torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                )

        def forward(
            self,
            embeds,
            prefix_mask,
            prefix_positions,
            noise,
            time,
            action_positions,
            action_mask,
        ):
            _, k, v = ops.llm_prefill(
                embeds,
                prefix_mask,
                prefix_positions,
                self.prefill,
                self.prefill_metadata,
            )
            return ops.action_expert(
                noise,
                time,
                k,
                v,
                action_positions,
                action_mask,
                self.action,
                self.action_metadata,
            )

    args = (
        torch.randn(1, 3, 2),
        torch.zeros(1, 1, 3, 3),
        torch.arange(3).unsqueeze(0),
        torch.randn(1, 2, 2),
        torch.ones(1),
        torch.arange(2).unsqueeze(0),
        torch.zeros(1, 1, 2, 5),
    )
    exported = torch.export.export(PrefixAndAction(), args)
    path = tmp_path / "pi05_step.pte"
    save(
        exported,
        str(path),
        output_format="executorch",
        partitioners=[EdgeLLMPartitioner()],
    )
    restored = deserialize_pte_binary(path.read_bytes()).program
    plan = restored.execution_plan[0]
    assert [delegate.id for delegate in plan.delegates] == [
        "EdgeLLMBackend",
        "EdgeLLMBackend",
    ]
    roles = []
    for delegate in plan.delegates:
        _, spec = deserialize_edge_component(
            restored.backend_delegate_data[delegate.processed.index].data
        )
        roles.append(spec.runner)
    assert roles == ["pi05_prefill", "pi05_action"]
    assert len(plan.inputs) == 7
