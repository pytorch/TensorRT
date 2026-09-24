from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from exporters.quantization import (
    FP8CheckpointLinear,
    ModelOptCheckpoint,
    load_modelopt_fp8_model,
)
from safetensors.torch import save_file


class TinyModel(nn.Module):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__()
        self.quant = nn.Linear(4, 3, bias=True, device=device)
        self.norm = nn.LayerNorm(3, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.quant(x))


def _write_checkpoint(path) -> dict[str, torch.Tensor]:
    tensors = {
        "quant.weight": torch.tensor(
            [
                [1.0, -1.0, 0.5, 0.25],
                [-0.5, 1.0, -0.25, 0.5],
                [0.25, 0.5, 1.0, -1.0],
            ],
            dtype=torch.float8_e4m3fn,
        ),
        "quant.bias": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float16),
        "quant.weight_quantizer._scale": torch.tensor(0.125, dtype=torch.float32),
        "quant.input_quantizer._amax": torch.tensor(56.0, dtype=torch.float32),
        "norm.weight": torch.ones(3, dtype=torch.float16),
        "norm.bias": torch.zeros(3, dtype=torch.float16),
    }
    shard = "model.safetensors"
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )
    return tensors


@pytest.mark.unit
def test_modelopt_checkpoint_indexes_quantized_linears(tmp_path):
    tensors = _write_checkpoint(tmp_path)
    checkpoint = ModelOptCheckpoint(tmp_path)

    assert checkpoint.quantized_linear_names() == ("quant",)
    torch.testing.assert_close(
        checkpoint.get_tensor("quant.weight"),
        tensors["quant.weight"],
    )


@pytest.mark.unit
def test_load_modelopt_checkpoint_replaces_linear_and_exports(tmp_path):
    tensors = _write_checkpoint(tmp_path)
    model = TinyModel(device="meta")
    stats = load_modelopt_fp8_model(
        model,
        tmp_path,
        device="cpu",
        dtype=torch.float16,
    )

    assert stats["fp8_linears"] == 1
    assert isinstance(model.quant, FP8CheckpointLinear)
    assert model.quant.weight.dtype == torch.float8_e4m3fn
    assert not any(parameter.is_meta for parameter in model.parameters())

    x = torch.tensor([[0.25, -0.5, 0.75, 1.0]], dtype=torch.float16)
    exported = torch.export.export(model.eval(), (x,), strict=False)
    assert torch.ops.edge_export.fp8_linear.default in {
        node.target for node in exported.graph.nodes
    }

    input_scale = tensors["quant.input_quantizer._amax"].float() / 448.0
    x_dq = (x.float() / input_scale).to(torch.float8_e4m3fn).to(x.dtype)
    x_dq = x_dq * input_scale.to(x.dtype)
    weight_dq = tensors["quant.weight"].to(x.dtype) * tensors[
        "quant.weight_quantizer._scale"
    ].to(x.dtype)
    expected = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(
            x_dq,
            weight_dq,
            tensors["quant.bias"],
        ),
        (3,),
        tensors["norm.weight"],
        tensors["norm.bias"],
    )
    torch.testing.assert_close(model(x), expected)
