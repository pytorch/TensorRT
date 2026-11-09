# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import json
import logging
import mmap
import os
import platform
import struct
import subprocess
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
        m.base <= tensor.data_ptr() < m.base + m.size for m in weight_mapping._mappings
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


def test_missing_names_are_listed_or_raise_with_strict(tmp_path, caplog):
    _reference(tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = torch.nn.ModuleDict(
            {"proj": torch.nn.Linear(8, 8), "extra": torch.nn.Linear(2, 2)}
        )
    with pytest.raises(ValueError, match="not in"):
        map_weights(model, tmp_path / "w.safetensors")  # strict by default
    assert model["proj"].weight.is_meta  # nothing changed
    with caplog.at_level(
        logging.WARNING, logger="torch_tensorrt.dynamo.tools.weight_mapping"
    ):
        mapping = map_weights(model, tmp_path / "w.safetensors", strict=False)
    assert sorted(mapping.missing) == ["extra.bias", "extra.weight"]
    assert _mapped(model["proj"].weight) and model["extra"].weight.is_meta
    # The missing tensors stay on meta, with no values, and the warning says so.
    assert "stay on the meta device" in caplog.text and "extra.weight" in caplog.text
    assert "keep their values" not in caplog.text


def test_safetensors_state_dict_maps_with_strict(tmp_path, caplog):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(0)
    ref = torch.nn.Linear(8, 8)
    ref.register_buffer("inv_freq", torch.rand(4), persistent=False)
    # The usual checkpoint: written by the safetensors package, and state_dict() leaves inv_freq out.
    safetensors_torch.save_file(ref.state_dict(), str(tmp_path / "sd.safetensors"))
    with torch.device("meta"):
        model = torch.nn.Linear(8, 8)
        model.register_buffer("inv_freq", torch.empty(4), persistent=False)
    with caplog.at_level(
        logging.WARNING, logger="torch_tensorrt.dynamo.tools.weight_mapping"
    ):
        mapping = map_weights(model, tmp_path / "sd.safetensors", strict=True)
    assert mapping.missing == ["inv_freq"]  # non-persistent: not required
    torch.testing.assert_close(model.weight, ref.weight)
    # It stays on meta, with no values; compile refuses it only if the graph reads it.
    assert "stay on the meta device" in caplog.text and "inv_freq" in caplog.text


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


def test_a_damaged_file_fails_before_mapping(tmp_path):
    """The whole layout is checked from the header, also tensors the module doesn't map."""
    with torch.device("meta"):
        model = torch.nn.Module()
        model.register_buffer("w", torch.empty(2))
    w = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}

    def unused(shape, offsets):
        return {"dtype": "F32", "shape": shape, "data_offsets": offsets}

    damaged = [
        ({"w": w, "unused": unused([2], [8, 16])}, 12, "it is truncated"),
        ({"w": w}, 12, "something follows the last tensor"),
        ({"w": w, "unused": unused([2], [4, 12])}, 12, "no gap or overlap"),
        ({"w": w, "unused": unused([2], [12, 20])}, 20, "no gap or overlap"),
        ({"w": w, "unused": unused([3], [8, 16])}, 16, "don't hold F32"),
    ]
    for header, data_bytes, message in damaged:
        _write_raw(tmp_path / "bad.safetensors", header, b"\0" * data_bytes)
        with pytest.raises(ValueError, match=message):
            map_weights(model, tmp_path / "bad.safetensors")
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


class _Aliased(torch.nn.Module):
    def __init__(self, second=lambda w: w.detach()):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.arange(8.0))
        self.w2 = torch.nn.Parameter(second(self.w1))


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_distinct_tensors_sharing_a_storage_keep_sharing_it(tmp_path, device):
    ref = _Aliased()
    save_weights_for_mapping(ref, tmp_path / "w.safetensors")
    with torch.device(device):
        model = _Aliased()
    mapping = map_weights(model, tmp_path / "w.safetensors")
    assert mapping.missing == [] and mapping.mapped_bytes == ref.w1.nbytes
    assert model.w1 is not model.w2
    with torch.no_grad():
        model.w1 += 1
    torch.testing.assert_close(model.w2, ref.w1 + 1)


def test_tensors_partly_sharing_a_storage_raise(tmp_path):
    model = _Aliased(second=lambda w: w.detach()[2:])
    with pytest.raises(ValueError, match="w2 and w1 share memory"):
        save_weights_for_mapping(model, tmp_path / "w.safetensors")
    _write_raw_tensors(
        tmp_path / "w.safetensors", {"w1": torch.zeros(8), "w2": torch.zeros(6)}
    )
    with pytest.raises(ValueError, match="w2 and w1 share memory"):
        map_weights(model, tmp_path / "w.safetensors")
    assert model.w2.data_ptr() == model.w1.data_ptr() + 8  # unchanged


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


def test_real_tensors_the_checkpoint_stores_are_read_from_it(tmp_path, caplog):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch.manual_seed(0)
    ref = torch.nn.BatchNorm1d(4)
    ref(torch.randn(16, 4) * 3 + 2)  # trains the running statistics
    safetensors_torch.save_file(ref.state_dict(), str(tmp_path / "sd.safetensors"))
    # Real buffers with their initial values, as accelerate's init_empty_weights() leaves them.
    model = torch.nn.BatchNorm1d(4)
    model.register_buffer("scale", torch.full((4,), 2.0))  # persistent, not stored
    source = weight_mapping.checkpoint_source(tmp_path / "sd.safetensors")
    with caplog.at_level(
        logging.WARNING, logger="torch_tensorrt.dynamo.tools.weight_mapping"
    ):
        save_weights_for_mapping(model, tmp_path / "w.safetensors", source=source)
    assert "not in the source" in caplog.text and "scale" in caplog.text
    with torch.device("meta"):
        mapped = torch.nn.BatchNorm1d(4)
        mapped.register_buffer("scale", torch.empty(4))
    map_weights(mapped, tmp_path / "w.safetensors")
    for name, value in ref.state_dict().items():
        torch.testing.assert_close(mapped.state_dict()[name], value)
    torch.testing.assert_close(mapped.scale, torch.full((4,), 2.0))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_saving_never_writes_through_a_link(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    path = tmp_path / "w.safetensors"
    (tmp_path / "w.safetensors.partial").symlink_to(
        victim
    )  # at the old fixed temporary name
    _reference(path)
    assert victim.read_text() == "keep"
    assert not path.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "victim.txt",
        "w.safetensors",
        "w.safetensors.partial",
    ]


class _TwoBuffers(torch.nn.Module):
    def __init__(
        self, n=mmap.PAGESIZE
    ):  # float32: 4 pages each, whatever the page size
        super().__init__()
        self.register_buffer("a", torch.arange(n, dtype=torch.float32))
        self.register_buffer("b", -torch.arange(n, dtype=torch.float32))


# MADV_PAGEOUT takes pages from the kernel's LRU lists, which new pages join in batches, so a count of the
# pages it drops needs more than a few pages.
_PAGEOUT_N = 2**18


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


def _present(tensor):
    page = mmap.PAGESIZE
    start = tensor.data_ptr() // page * page
    end = -(-(tensor.data_ptr() + tensor.nbytes) // page) * page
    return weight_mapping._present_bytes(start, end - start)


@pytest.mark.skipif(not _has_pageout(), reason="needs MADV_PAGEOUT (Linux 5.4+)")
def test_release_keeps_writes(tmp_path):
    ref, model = _mapped_two_buffers(tmp_path)
    assert release_mapped_storage(model.b.untyped_storage()) is not None
    torch.testing.assert_close(model.a, ref.a + 5.0)
    torch.testing.assert_close(model.b, ref.b + 5.0)  # swapped out, not discarded
    assert release_mapped_storage(torch.ones(4).untyped_storage()) is None


@pytest.mark.skipif(not hasattr(mmap, "MADV_DONTNEED"), reason="needs MADV_DONTNEED")
def test_release_without_pageout_drops_only_its_own_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_mapping, "_MADV_PAGEOUT", None)
    ref, model = _mapped_two_buffers(tmp_path)
    assert release_mapped_storage(model.b.untyped_storage()) > 0
    torch.testing.assert_close(model.a, ref.a + 5.0)  # a neighbour's writes survive
    # b's whole pages were re-read from the file; its edge page, shared with a neighbour, was kept.
    assert not torch.equal(model.b, ref.b + 5.0)


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux's page map")
def test_release_drops_pages_of_a_file_the_process_cannot_write(tmp_path, monkeypatch):
    """MADV_PAGEOUT skips a file's pages, and still succeeds, unless the process owns the file or may write
    to it: a model file baked into a container image as root and read by a normal user, say.
    """
    ref = _TwoBuffers()
    save_weights_for_mapping(ref, tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _TwoBuffers()
    with monkeypatch.context() as m:
        m.setattr(os, "geteuid", lambda: 12345)  # neither root nor the owner
        m.setattr(os, "access", lambda *args, **kwargs: False)
        map_weights(model, tmp_path / "w.safetensors")
    assert not weight_mapping._mappings[-1].can_pageout
    torch.testing.assert_close(model.a, ref.a)  # reads its pages in
    assert release_mapped_storage(model.a.untyped_storage()) > 0
    assert _present(model.a) <= 2 * mmap.PAGESIZE  # only the edge pages it shares


@pytest.mark.skipif(not _has_pageout(), reason="needs MADV_PAGEOUT (Linux 5.4+)")
def test_release_counts_only_pages_that_left_memory(tmp_path):
    ref = _TwoBuffers(_PAGEOUT_N)
    save_weights_for_mapping(ref, tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _TwoBuffers(_PAGEOUT_N)
    map_weights(model, tmp_path / "w.safetensors")
    torch.testing.assert_close(model.a, ref.a)  # reads its pages in
    # MADV_PAGEOUT leaves pages that another process maps too, yet succeeds.
    reader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import mmap, sys\n"
            "f = open(sys.argv[1], 'rb')\n"
            "m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)\n"
            "sum(m[i] for i in range(0, len(m), mmap.PAGESIZE))\n"
            "print(flush=True)\n"
            "sys.stdin.read()",
            str(tmp_path / "w.safetensors"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        reader.stdout.readline()
        assert release_mapped_storage(model.a.untyped_storage()) == 0
    finally:
        reader.communicate()
    assert release_mapped_storage(model.a.untyped_storage()) > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's memory accounting")
def test_a_file_larger_than_memory_maps(tmp_path):
    """A copy-on-write mapping reserves memory for every page by default, and Linux refuses one larger than
    RAM and swap: the models too big to load are the ones to map."""
    with open("/proc/sys/vm/overcommit_memory") as f:
        if f.read().strip() == "2":
            pytest.skip("strict overcommit refuses it anyway")
    with open("/proc/meminfo") as f:
        meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    numel = 2 * (meminfo["MemTotal"] + meminfo["SwapTotal"]) // 4
    header = {"w": {"dtype": "F32", "shape": [numel], "data_offsets": [0, numel * 4]}}
    path = tmp_path / "big.safetensors"
    _write_raw(path, header)
    os.truncate(path, path.stat().st_size + numel * 4)  # sparse: takes no disk
    with torch.device("meta"):
        model = torch.nn.Module()
        model.register_buffer("w", torch.empty(numel))
    map_weights(model, path)
    assert model.w[-1].item() == 0.0
    del model


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


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux's page map")
def test_release_storage_drops_pages_of_mapped_weights(tmp_path):
    ref = _TwoBuffers(_PAGEOUT_N)
    save_weights_for_mapping(ref, tmp_path / "w.safetensors")
    with torch.device("meta"):
        model = _TwoBuffers(_PAGEOUT_N)
    map_weights(model, tmp_path / "w.safetensors")
    torch.testing.assert_close(model.a, ref.a)  # reads its pages in
    mapped = model.a.untyped_storage()
    # Its whole pages are dropped, not resized away.
    assert 0 < release_storage(mapped) < mapped.nbytes() == ref.a.nbytes
    torch.testing.assert_close(model.a, ref.a)
    plain = torch.ones(4).untyped_storage()
    assert release_storage(plain, mapped_only=True) == 0 and plain.nbytes() == 16
    assert release_storage(plain) == 16 and plain.nbytes() == 0  # resize_(0) as before


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_refuses_a_weights_file_changed_after_mapping(tmp_path):
    """Mapped weights are read during compilation, so their file must still be the one mapped: reading a
    weight past the end of a file truncated since would kill the process with SIGBUS."""
    path = tmp_path / "w.safetensors"
    _reference(path)
    with torch.device("meta"):
        model = _Tied().eval()
    map_weights(model, path)
    x = torch.randn(3, 8)
    ep = torch.export.export(model, (x,))
    stat = path.stat()
    # Rewritten in place: same size, later modification time.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="changed after map_weights mapped it"):
        torch_tensorrt.dynamo.compile(ep, inputs=[x.cuda()], min_block_size=1)
    path.unlink()
    with pytest.raises(ValueError, match="was removed after map_weights mapped it"):
        torch_tensorrt.dynamo.compile(ep, inputs=[x.cuda()], min_block_size=1)


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
    assert calls and None not in calls  # every released weight was a mapped one
    torch.testing.assert_close(
        model.proj.weight, ref.proj.weight
    )  # re-read from the file
