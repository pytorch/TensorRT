# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import (
    map_weights,
    release_mapped_storage,
    save_weights_for_mapping,
)


class _Tied(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8, dtype=dtype)
        self.proj = torch.nn.Linear(8, 8, dtype=dtype)
        self.head = torch.nn.Linear(8, 16, bias=False, dtype=dtype)
        self.head.weight = self.embed.weight
        self.register_buffer("scale", torch.full((8,), 2.0, dtype=dtype))

    def forward(self, x):
        return self.head(self.proj(x) * self.scale)


def _reference(path, dtype=torch.float32):
    torch.manual_seed(0)
    model = _Tied(dtype).eval()
    save_weights_for_mapping(model, path)
    return model


def _all_mapped(model, mapping):
    return all(mapping.contains(t.data_ptr()) for t in model.state_dict().values())


def test_meta_module_maps_without_allocating(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied().eval()
    mapping = map_weights(model, tmp_path / "w.safetensors")
    assert _all_mapped(model, mapping)
    assert model.head.weight is model.embed.weight
    assert not mapping.copied and not mapping.missing
    x = torch.randn(3, 8)
    torch.testing.assert_close(model(x), ref(x))
    mapping.close()


def test_real_module_keeps_its_parameter_objects(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    model = _Tied().eval()
    weight = model.proj.weight
    mapping = map_weights(model, tmp_path / "w.safetensors")
    assert model.proj.weight is weight and _all_mapped(model, mapping)
    torch.testing.assert_close(model.proj.weight, ref.proj.weight)
    mapping.close()


def test_key_map_and_missing_names(tmp_path):
    _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = torch.nn.ModuleDict({"inner": _Tied(), "extra": torch.nn.Linear(2, 2)})
    mapping = map_weights(
        model, tmp_path / "w.safetensors", key_map=lambda n: n.removeprefix("inner.")
    )
    assert _all_mapped(model["inner"], mapping)
    assert sorted(mapping.still_meta) == ["extra.bias", "extra.weight"]
    assert model["extra"].weight.is_meta
    mapping.close()


def test_dtype_mismatch_casts_or_raises(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied(torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        map_weights(model, tmp_path / "w.safetensors", strict=True)
    mapping = map_weights(model, tmp_path / "w.safetensors")
    assert set(mapping.copied) >= {"proj.weight", "scale"} and mapping.mapped_bytes == 0
    torch.testing.assert_close(model.proj.weight, ref.proj.weight.bfloat16())
    mapping.close()


def test_shape_mismatch_raises(tmp_path):
    _reference(tmp_path / "w.safetensors")
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(4, 8)
    with pytest.raises(ValueError, match="shape"):
        map_weights(model, tmp_path / "w.safetensors")


def test_bf16_checkpoint_round_trip(tmp_path):
    ref = _reference(tmp_path / "w.safetensors", torch.bfloat16)
    with torch.device("meta"):
        model = _Tied(torch.bfloat16)
    mapping = map_weights(model, tmp_path / "w.safetensors", strict=True)
    assert _all_mapped(model, mapping)
    for name, tensor in ref.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name
    mapping.close()


def test_exported_program_state_is_remapped(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    x = torch.randn(3, 8)
    ep = torch.export.export(_Tied().eval(), (x,))
    mapping = map_weights(ep, tmp_path / "w.safetensors")
    assert all(mapping.contains(t.data_ptr()) for t in ep.state_dict.values())
    torch.testing.assert_close(ep.module()(x), ref(x))
    mapping.close()


def test_release_drops_pages_and_rereads_from_file(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied()
    mapping = map_weights(model, tmp_path / "w.safetensors")
    storage = model.proj.weight.untyped_storage()
    assert release_mapped_storage(storage)
    torch.testing.assert_close(model.proj.weight, ref.proj.weight)
    assert not release_mapped_storage(torch.ones(4).untyped_storage())
    mapping.close()
    assert not release_mapped_storage(storage)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_from_mapped_cpu_weights(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied().eval()
    mapping = map_weights(model, tmp_path / "w.safetensors")
    x = torch.randn(3, 8)
    ep = torch.export.export(model, (x,))
    trt = torch_tensorrt.dynamo.compile(ep, inputs=[x.cuda()], min_block_size=1)
    torch.testing.assert_close(trt(x.cuda()).cpu(), ref(x), rtol=1e-3, atol=1e-3)
    mapping.close()
