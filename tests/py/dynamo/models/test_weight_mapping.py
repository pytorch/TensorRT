# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import json
import mmap
import platform
import struct
import sys

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import _weight_release
from torch_tensorrt.dynamo._weight_release import release_storage
from torch_tensorrt.dynamo.tools import (
    map_weights,
    save_weights_for_mapping,
    weight_mapping,
)
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage


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


def _mapped(tensor):
    return any(
        base <= tensor.data_ptr() < base + size
        for base, size, _ in weight_mapping._mappings
    )


def _write_raw(path, header, data=b""):
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_meta_module_maps_without_allocating(tmp_path, dtype):
    ref = _reference(tmp_path / "w.safetensors", dtype)
    with torch.device("meta"):
        model = _Tied(dtype).eval()
    mapping = map_weights(model, tmp_path / "w.safetensors", strict=True)
    assert all(_mapped(t) for t in model.state_dict().values())
    assert model.head.weight is model.embed.weight  # tied weights stay tied
    assert not mapping.missing
    assert (
        mapping.mapped_bytes
        == sum(t.nbytes for t in ref.state_dict().values()) - ref.head.weight.nbytes
    )
    for name, tensor in ref.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor), name


def test_real_module_keeps_its_parameter_objects(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    model = _Tied().eval()
    weight = model.proj.weight
    map_weights(model, tmp_path / "w.safetensors")
    assert model.proj.weight is weight
    assert all(_mapped(t) for t in model.state_dict().values())
    torch.testing.assert_close(model.proj.weight, ref.proj.weight)


def test_a_meta_module_is_written_from_a_checkpoint_in_chunks(tmp_path):
    """A model built on meta, in other dtypes than its float32 checkpoint and under other names, is
    written from the checkpoint chunk by chunk, and the file maps back to the checkpoint's values.
    """
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    torch.manual_seed(0)
    reference = _Tied(torch.float32).eval()
    checkpoint = str(tmp_path / "checkpoint.safetensors")
    save_file(
        {
            f"model.{k}": v
            for k, v in reference.state_dict().items()
            if k != "head.weight"
        },
        checkpoint,
    )
    with torch.device("meta"):
        model = _Tied(torch.bfloat16)
    chunks = []
    read = weight_mapping.checkpoint_source(
        checkpoint, lambda name: f"model.{name}", chunk_bytes=64
    )

    def source(name, like):
        for chunk in read(name, like):
            chunks.append(chunk.numel())
            yield chunk

    path = str(tmp_path / "w.safetensors")
    save_weights_for_mapping(model, path, source=source)
    assert max(chunks) * 4 <= 64  # never a whole tensor: 64-byte chunks of float32
    mapped = _Tied(torch.bfloat16)
    map_weights(mapped, path, strict=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(mapped.state_dict()[name], value.to(torch.bfloat16))
    assert mapped.head.weight is mapped.embed.weight


def test_a_source_of_the_wrong_size_raises_and_leaves_no_file(tmp_path):
    with torch.device("meta"):
        model = torch.nn.Linear(4, 4)
    path = tmp_path / "w.safetensors"
    with pytest.raises(ValueError, match="bytes, expected"):
        save_weights_for_mapping(
            model, path, source=lambda name, like: torch.zeros(2, *like.shape[1:])
        )
    assert not path.exists() and not (tmp_path / "w.safetensors.partial").exists()
    with pytest.raises(ValueError, match="meta device"):
        save_weights_for_mapping(model, path)


def test_a_submodule_maps_from_the_whole_model_with_a_prefix(tmp_path):
    path = str(tmp_path / "w.safetensors")
    model = _reference(path)
    proj = torch.nn.Linear(8, 8, device="meta")
    result = map_weights(proj, path, prefix="proj.", strict=True)
    assert result.missing == []
    assert torch.equal(proj.weight, model.proj.weight)
    assert torch.equal(proj.bias, model.proj.bias)


def test_missing_names_are_listed_or_raise_with_strict(tmp_path):
    _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = torch.nn.ModuleDict(
            {"proj": torch.nn.Linear(8, 8), "extra": torch.nn.Linear(2, 2)}
        )
    with pytest.raises(ValueError, match="not in"):
        map_weights(model, tmp_path / "w.safetensors")  # strict by default
    assert model["proj"].weight.is_meta  # nothing changed
    mapping = map_weights(model, tmp_path / "w.safetensors", strict=False)
    assert sorted(mapping.missing) == ["extra.bias", "extra.weight"]
    assert _mapped(model["proj"].weight) and model["extra"].weight.is_meta


def test_safetensors_state_dict_maps_with_strict(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(0)
    ref = torch.nn.Linear(8, 8)
    ref.register_buffer("inv_freq", torch.rand(4), persistent=False)
    # The usual checkpoint: written by the safetensors package, and state_dict() leaves inv_freq out.
    safetensors_torch.save_file(ref.state_dict(), str(tmp_path / "sd.safetensors"))
    with torch.device("meta"):
        model = torch.nn.Linear(8, 8)
        model.register_buffer("inv_freq", torch.empty(4), persistent=False)
    mapping = map_weights(model, tmp_path / "sd.safetensors", strict=True)
    assert mapping.missing == ["inv_freq"]  # non-persistent: not required
    torch.testing.assert_close(model.weight, ref.weight)


def test_a_mismatch_raises_and_changes_nothing(tmp_path):
    _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        bf16 = _Tied(torch.bfloat16)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(8, 8)  # matches, and is resolved first
        model.register_buffer("scale", torch.empty(4))  # the file has [8]
    with pytest.raises(ValueError, match="dtype"):
        map_weights(bf16, tmp_path / "w.safetensors")
    with pytest.raises(ValueError, match="shape"):
        map_weights(model, tmp_path / "w.safetensors")
    assert bf16.proj.weight.is_meta and model.proj.weight.is_meta


def test_malformed_headers_raise(tmp_path):
    with torch.device("meta"):
        model = torch.nn.Module()
        model.register_buffer("w", torch.empty(4, 4))
    bad_entries = (
        {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 0]},  # too short
        {"dtype": "F32", "shape": [4, 4], "data_offsets": [-16, 48]},  # before the data
        {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},  # past the end
    )
    for entry in bad_entries:
        _write_raw(tmp_path / "bad.safetensors", {"w": entry}, b"\0" * 32)
        with pytest.raises(ValueError, match="data_offsets"):
            map_weights(model, tmp_path / "bad.safetensors")
    (tmp_path / "short.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(ValueError, match="header length"):
        map_weights(model, tmp_path / "short.safetensors")
    assert model.w.is_meta


def test_a_tie_stored_under_its_later_name_maps(tmp_path):
    torch.manual_seed(0)
    ref = _Tied().eval()
    weights = {k: v for k, v in ref.state_dict().items() if k != "embed.weight"}
    _write_raw_tensors(tmp_path / "w.safetensors", weights)  # head.weight only
    with torch.device("meta"):
        model = _Tied().eval()
    mapping = map_weights(model, tmp_path / "w.safetensors")
    assert mapping.missing == []
    assert model.head.weight is model.embed.weight
    torch.testing.assert_close(model.embed.weight, ref.embed.weight)


def _write_raw_tensors(path, tensors):
    header, data, offset = {}, b"", 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        dtype = weight_mapping._DTYPE_NAMES[t.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data, offset = data + raw, offset + len(raw)
    _write_raw(path, header, data)


def test_unsupported_dtypes_and_sharded_indexes_raise_clearly(tmp_path):
    with torch.device("meta"):
        model = torch.nn.Module()
        model.register_buffer("w", torch.empty(4))
    _write_raw(
        tmp_path / "q.safetensors",
        {"w": {"dtype": "Q4", "shape": [4], "data_offsets": [0, 2]}},
        b"\0\0",
    )
    with pytest.raises(ValueError, match="dtype Q4 is not supported"):
        map_weights(model, tmp_path / "q.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(ValueError, match="index of a sharded checkpoint"):
        map_weights(model, tmp_path / "model.safetensors.index.json")


def test_a_sharded_checkpoint_converts_through_its_index(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(0)
    ref = _Tied().eval()
    state = {k: v for k, v in ref.state_dict().items() if k != "head.weight"}
    names = sorted(state)
    shards = {"a.safetensors": names[:2], "b.safetensors": names[2:]}
    for file, keys in shards.items():
        safetensors_torch.save_file({k: state[k] for k in keys}, str(tmp_path / file))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {k: f for f, ks in shards.items() for k in ks}})
    )
    with torch.device("meta"):
        model = _Tied(torch.bfloat16).eval()
    path = tmp_path / "w.safetensors"
    save_weights_for_mapping(
        model, path, source=weight_mapping.checkpoint_source(index)
    )
    mapped = _Tied(torch.bfloat16)
    map_weights(mapped, path)
    for name, value in ref.state_dict().items():
        torch.testing.assert_close(mapped.state_dict()[name], value.to(torch.bfloat16))


def test_a_buffer_the_checkpoint_does_not_store_must_be_real(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    ref = torch.nn.Linear(8, 8)
    ref.register_buffer("inv_freq", torch.rand(4), persistent=False)
    safetensors_torch.save_file(ref.state_dict(), str(tmp_path / "sd.safetensors"))
    with torch.device("meta"):
        model = torch.nn.Linear(8, 8)
        model.register_buffer("inv_freq", torch.empty(4), persistent=False)
    source = weight_mapping.checkpoint_source(tmp_path / "sd.safetensors")
    with pytest.raises(ValueError, match="inv_freq is not in .*init_empty_weights"):
        save_weights_for_mapping(model, tmp_path / "w.safetensors", source=source)


class _TwoBuffers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n = mmap.PAGESIZE  # float32: 4 pages each, whatever the page size
        self.register_buffer("a", torch.arange(n, dtype=torch.float32))
        self.register_buffer("b", -torch.arange(n, dtype=torch.float32))


def _mapped_two_buffers(tmp_path):
    ref = _TwoBuffers()
    save_weights_for_mapping(ref, tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _TwoBuffers()
    map_weights(model, tmp_path / "w.safetensors")
    model.a.add_(5.0)
    model.b.add_(5.0)
    return ref, model


def _has_pageout():
    release = tuple(int(p) for p in platform.release().split("-")[0].split(".")[:2])
    return sys.platform == "linux" and release >= (5, 4)


@pytest.mark.skipif(not _has_pageout(), reason="needs MADV_PAGEOUT (Linux 5.4+)")
def test_release_keeps_writes(tmp_path):
    ref, model = _mapped_two_buffers(tmp_path)
    assert release_mapped_storage(model.b.untyped_storage())
    torch.testing.assert_close(model.a, ref.a + 5.0)
    torch.testing.assert_close(model.b, ref.b + 5.0)  # swapped out, not discarded
    assert not release_mapped_storage(torch.ones(4).untyped_storage())


@pytest.mark.skipif(not hasattr(mmap, "MADV_DONTNEED"), reason="needs MADV_DONTNEED")
def test_release_without_pageout_drops_only_its_own_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_mapping, "_PAGE_DROP_ADVICE", (mmap.MADV_DONTNEED,))
    ref, model = _mapped_two_buffers(tmp_path)
    assert release_mapped_storage(model.b.untyped_storage())
    torch.testing.assert_close(model.a, ref.a + 5.0)  # a neighbour's writes survive
    # b's whole pages were re-read from the file; its edge page, shared with a neighbour, was kept.
    assert not torch.equal(model.b, ref.b + 5.0)


def test_mappings_are_freed_with_their_views(tmp_path):
    _reference(tmp_path / "w.safetensors")
    gc.collect()
    release_mapped_storage(torch.ones(1).untyped_storage())  # prunes dead entries
    before = len(weight_mapping._mappings)
    for _ in range(3):
        with torch.device("meta"):
            model = _Tied()
        map_weights(model, tmp_path / "w.safetensors")
        del model
    gc.collect()
    release_mapped_storage(torch.ones(1).untyped_storage())
    assert len(weight_mapping._mappings) == before


def test_release_storage_drops_pages_of_mapped_weights(tmp_path):
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied()
    map_weights(model, tmp_path / "w.safetensors")
    mapped = model.proj.weight.untyped_storage()
    assert release_storage(mapped) == mapped.nbytes() > 0  # pages dropped, not resized
    torch.testing.assert_close(model.proj.weight, ref.proj.weight)
    plain = torch.ones(4).untyped_storage()
    assert release_storage(plain) == 16 and plain.nbytes() == 0  # resize_(0) as before


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_mapped_weights_and_release_them(tmp_path, monkeypatch):
    calls = []

    def spy(storage):
        calls.append(release_mapped_storage(storage))
        return calls[-1]

    monkeypatch.setattr(_weight_release, "release_mapped_storage", spy)
    ref = _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _Tied().eval()
    map_weights(model, tmp_path / "w.safetensors", strict=True)
    x = torch.randn(3, 8)
    ep = torch.export.export(model, (x,))
    trt = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[x.cuda()],
        min_block_size=1,
        release_weights_after_build=True,
        disable_tf32=True,  # TF32 matmuls differ from eager by ~2e-3
    )
    # TensorRT-RTX differs from eager by up to 2e-3 even without TF32.
    torch.testing.assert_close(trt(x.cuda()).cpu(), ref(x), rtol=1e-2, atol=1e-2)
    assert calls and all(calls)  # every released weight was a mapped one
    torch.testing.assert_close(
        model.proj.weight, ref.proj.weight
    )  # re-read from the file
