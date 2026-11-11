# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import errno
import gc
import inspect
import logging
import os
import platform

import pytest
import torch
import torch_tensorrt
from safetensors.torch import save_file
from torch.multiprocessing.reductions import StorageWeakRef
from torch_tensorrt._utils import _cache_root
from torch_tensorrt.dynamo import _low_memory, low_memory
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.tools import map_weights
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage

needs_file_deletion = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="low_memory raises NotImplementedError on Windows",
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.full((8,), 2.0), persistent=False)

    def forward(self, x):
        return self.proj(x) * self.scale


def _loaded():
    torch.manual_seed(0)
    return _Model().eval()


def _tensors(model):
    return [*model.parameters(), *model.buffers()]


def test_compile_settings_hold_the_given_settings_and_the_defaults():
    # The weights are written only on entering the block.
    assert dict(low_memory(_loaded()).compile_settings) == {
        "offload_module_to_cpu": True,
        "enable_resource_partitioning": True,
    }
    settings = low_memory(
        _loaded(),
        offload_module_to_cpu=False,
        cpu_memory_budget=3 << 30,
        release_weights_after_build=0,
    ).compile_settings
    assert dict(settings) == {
        "offload_module_to_cpu": False,
        "enable_resource_partitioning": True,
        "cpu_memory_budget": 3 << 30,
        "release_weights_after_build": False,
    }
    assert (
        CompilationSettings(release_weights_after_build=0).release_weights_after_build
        is False
    )


def test_the_settings_are_compile_settings():
    # compile takes **kwargs, so a renamed setting would be ignored there, not rejected.
    block = {
        name
        for name, p in inspect.signature(low_memory).parameters.items()
        if p.kind is p.KEYWORD_ONLY
    }
    assert block and block <= set(
        inspect.signature(torch_tensorrt.dynamo.compile).parameters
    )


def test_bad_arguments_raise():
    with pytest.raises(TypeError, match="optimisation_level"):
        low_memory(_loaded(), optimisation_level=1)
    # Only settings that lower compilation memory: the rest go to compile itself.
    with pytest.raises(TypeError, match="min_block_size"):
        low_memory(_loaded(), min_block_size=1)
    # The loaded model, not a function that builds it, nor a weights file (the 2.15 form).
    with pytest.raises(TypeError, match="takes the loaded model"):
        low_memory(_Model)
    with pytest.raises(TypeError, match="takes the loaded model"):
        low_memory(lambda: _Model())
    with pytest.raises(TypeError, match="positional"):
        low_memory(_Model, "w.safetensors")


def test_entering_on_windows_raises(monkeypatch):
    monkeypatch.setattr(_low_memory.platform, "system", lambda: "Windows")
    with pytest.raises(NotImplementedError, match="Windows can't delete"):
        with low_memory(_loaded()):
            pass


@needs_file_deletion
def test_entering_moves_the_weights_into_a_file_and_frees_the_loaded_copy(caplog):
    model = _loaded()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model(x)
    loaded = [StorageWeakRef(t.untyped_storage()) for t in _tensors(model)]
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory(model) as low_mem:
            assert low_mem.model is model
            assert low_mem.weights_file.parent == _cache_root() / "offloaded_weights"
            assert low_mem.weights_file.is_file()
            gc.collect()
            assert all(ref.expired() for ref in loaded)
            # Every parameter and buffer, the non-persistent one too, is a view of the file.
            assert all(
                release_mapped_storage(t.untyped_storage()) is not None
                for t in _tensors(model)
            )
            assert low_mem.report.mapped_bytes == sum(t.nbytes for t in _tensors(model))
            with torch.no_grad():
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert "still in memory" not in caplog.text


@needs_file_deletion
def test_the_file_is_deleted_once_the_model_is():
    model = _loaded()
    with low_memory(model) as low_mem:
        path = low_mem.weights_file
    assert path.is_file()  # the model still reads it after the block
    model(torch.randn(3, 8))
    del model, low_mem
    gc.collect()
    assert not path.exists()


@needs_file_deletion
def test_weights_held_outside_the_model_are_reported(caplog):
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512)
    )
    held = model(
        torch.randn(3, 512)
    )  # gradients on: the graph holds the last layer's weight
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory(model):
            pass
    assert "1.0 MiB of the model's loaded weights are still in memory" in caplog.text
    del held


@needs_file_deletion
def test_a_model_on_meta_raises():
    with torch.device("meta"):
        model = _Model()
    with pytest.raises(ValueError, match="meta device.*proj.weight"):
        with low_memory(model):
            pass


@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_model_on_the_gpu_moves_to_the_cpu():
    model = _loaded().cuda()
    x = torch.randn(3, 8, device="cuda")
    with torch.no_grad():
        expected = model(x)
    with low_memory(model):
        assert not any(t.is_cuda for t in _tensors(model))
        with torch.no_grad():
            torch.testing.assert_close(model(x.cpu()), expected.cpu())


@needs_file_deletion
def test_a_failed_write_names_tmpdir_and_leaves_the_model_loaded(monkeypatch):
    model = _loaded()
    weight = model.proj.weight.untyped_storage().data_ptr()

    def disk_full(module, path):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_low_memory, "save_weights_for_mapping", disk_full)
    with pytest.raises(OSError, match="No space left on device.*TMPDIR"):
        with low_memory(model):
            pass
    assert model.proj.weight.untyped_storage().data_ptr() == weight


@needs_file_deletion
def test_a_failed_mapping_deletes_the_file(monkeypatch):
    paths = []
    new_path = _low_memory._new_weights_file_path
    monkeypatch.setattr(
        _low_memory,
        "_new_weights_file_path",
        lambda: paths.append(new_path()) or paths[-1],
    )

    def fail(module, path):
        assert os.path.isfile(path)
        raise RuntimeError("cannot map")

    monkeypatch.setattr(_low_memory, "map_weights", fail)
    with pytest.raises(RuntimeError, match="cannot map"):
        with low_memory(_loaded()):
            pass
    assert paths and not paths[0].exists()


@needs_file_deletion
def test_files_of_processes_that_no_longer_run_are_deleted():
    directory = _cache_root() / "offloaded_weights"
    directory.mkdir(parents=True, exist_ok=True)
    # No process can have an id above the kernel's limit.
    gone = int(open("/proc/sys/kernel/pid_max").read()) + 1
    left_behind = [
        directory / f"weights_{gone}_test.safetensors",
        directory / f"weights_{gone}_test.safetensors.0123.partial",
    ]
    mine = directory / f"weights_{os.getpid()}_test.safetensors"
    for path in [*left_behind, mine]:
        path.write_bytes(b"\0")
    try:
        with low_memory(_loaded()):
            pass
        assert not any(path.exists() for path in left_behind)
        assert mine.exists()
    finally:
        for path in [*left_behind, mine]:
            path.unlink(missing_ok=True)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_weight_the_graph_never_reads_may_stay_on_meta(tmp_path):
    class WithUnusedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 4)  # never called in forward

        def forward(self, x):
            return torch.relu(self.proj(x))

    torch.manual_seed(0)
    reference = WithUnusedHead().eval()
    proj_only = {
        k: v for k, v in reference.state_dict().items() if k.startswith("proj.")
    }
    save_file(proj_only, str(tmp_path / "proj.safetensors"))
    with torch.device("meta"):
        model = WithUnusedHead().eval()
    map_weights(model, tmp_path / "proj.safetensors", strict=False)
    assert model.head.weight.is_meta
    x = torch.randn(3, 8)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), inputs=[x.cuda()], min_block_size=1
    )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )


@needs_file_deletion
def test_leaving_drops_the_pages_also_on_an_exception(monkeypatch):
    model = _loaded()
    weight = model.proj.weight.detach().clone()
    dropped = []

    def spy(storage):
        dropped.append(release_mapped_storage(storage))
        return dropped[-1]

    monkeypatch.setattr(_low_memory, "release_mapped_storage", spy)
    low_mem = low_memory(model)
    with pytest.raises(RuntimeError, match="boom"):
        with low_mem:
            raise RuntimeError("boom")
    assert dropped and None not in dropped  # every one a mapped weight
    torch.testing.assert_close(low_mem.model.proj.weight, weight)


@needs_file_deletion
def test_a_block_that_keeps_weights_keeps_their_pages(monkeypatch):
    dropped = []
    monkeypatch.setattr(_low_memory, "release_mapped_storage", dropped.append)
    with low_memory(_loaded(), release_weights_after_build=False):
        pass
    assert not dropped


def test_a_program_with_weights_on_meta_does_not_compile():
    with torch.device("meta"):
        model = _Model().eval()
    exported = torch.export.export(model, (torch.randn(3, 8, device="meta"),))
    with pytest.raises(ValueError, match="meta device.*proj.weight"):
        torch_tensorrt.dynamo.compile(
            exported, inputs=[torch.randn(3, 8)], min_block_size=1
        )


def test_a_dry_run_accepts_weights_on_meta():
    with torch.device("meta"):
        model = _Model().eval()
    exported = torch.export.export(model, (torch.randn(3, 8, device="meta"),))
    torch_tensorrt.dynamo.compile(
        exported, inputs=[torch.randn(3, 8)], min_block_size=1, dryrun=True
    )


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_with_the_block_settings(monkeypatch):
    released = []
    real = torch_tensorrt.dynamo._weight_release.release_mapped_storage

    def spy(storage):
        released.append(real(storage))
        return released[-1]

    monkeypatch.setattr(
        torch_tensorrt.dynamo._weight_release, "release_mapped_storage", spy
    )
    x = torch.randn(3, 8)
    model = _loaded()
    with torch.no_grad():
        expected = model(x)
    with low_memory(model) as low_mem:
        exported = torch.export.export(low_mem.model, (x,))
        trt_model = torch_tensorrt.compile(
            exported,
            ir="dynamo",
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), expected, rtol=1e-3, atol=1e-3
    )
    # By default, mapped weights' pages are dropped.
    assert released and None not in released
    with torch.no_grad():
        torch.testing.assert_close(low_mem.model(x), expected)

    # With release_weights_after_build=False in the settings passed to compile, none are.
    released.clear()
    with low_memory(_loaded(), release_weights_after_build=False) as low_mem:
        exported = torch.export.export(low_mem.model, (x,))
        torch_tensorrt.compile(
            exported,
            ir="dynamo",
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    assert not released

    # An ordinary model's weights are left alone by default.
    plain = _Model().eval()
    exported = torch.export.export(plain, (x,))
    torch_tensorrt.dynamo.compile(exported, inputs=[x.cuda()], min_block_size=1)
    assert plain.proj.weight.untyped_storage().nbytes() > 0


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_block_does_not_change_other_compiles():
    """A model compiled inside the block without its settings keeps its own: a GPU model isn't offloaded."""
    gpu_model = _Model().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    with low_memory(_loaded()):
        torch_tensorrt.dynamo.compile(
            torch.export.export(gpu_model, (x,)),
            inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},
        )
    assert all(p.is_cuda for p in gpu_model.parameters())
    gpu_model(x)


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_pytorch_part_reading_a_mapped_weight_runs():
    x = torch.randn(3, 8)
    model = _loaded()
    with torch.no_grad():
        expected = model(x)
    with low_memory(model) as low_mem:
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(low_mem.model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},  # reads the mapped scale
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), expected, rtol=1e-3, atol=1e-3
    )


def _resident_mib(path):
    """MiB of the file at `path` resident in this process: its mappings' Rss in /proc/self/smaps."""
    total, in_file = 0, False
    for line in open("/proc/self/smaps"):
        fields = line.split()
        if fields and "-" in fields[0] and len(fields) >= 5:  # a mapping's header line
            in_file = fields[-1] == path
        elif in_file and fields[0] == "Rss:":
            total += int(fields[1])
    return total / 1024


class _Big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(512, 512, bias=False)  # 1 MiB of float32 weights

    def forward(self, x):
        return self.proj(x)


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists("/proc/self/smaps"),
    reason="needs CUDA and /proc/self/smaps",
)
def test_the_weights_file_leaves_memory():
    x = torch.randn(3, 512)
    with low_memory(_Big().eval()) as low_mem:
        path = os.path.realpath(low_mem.weights_file)
        with torch.no_grad():
            low_mem.model(x)  # reads every weight in
        assert _resident_mib(path) >= 0.9
        torch_tensorrt.dynamo.compile(
            torch.export.export(low_mem.model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
        # Read inside the block: leaving it drops the pages too, so a reading after it can't tell that
        # compile released them. Only the pages at the weight's ends, which may hold a neighbour's data,
        # can stay.
        assert _resident_mib(path) < 0.05
