# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import logging
import os

import pytest
import torch
import torch_tensorrt
from safetensors.torch import save_file
from torch_tensorrt.dynamo import _low_memory, low_memory
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.tools import map_weights, save_weights_for_mapping
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.full((8,), 2.0), persistent=False)

    def forward(self, x):
        return self.proj(x) * self.scale


def _weights(path):
    torch.manual_seed(0)
    reference = _Model().eval()
    save_weights_for_mapping(reference, path)
    return reference


def test_compile_settings_hold_the_given_settings_and_the_defaults():
    # The file is read only on entering the block.
    assert dict(low_memory(_Model, "w.safetensors").compile_settings) == {
        "offload_module_to_cpu": True,
        "enable_resource_partitioning": True,
    }
    settings = low_memory(
        _Model,
        "w.safetensors",
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
        low_memory(_Model, "w.safetensors", optimisation_level=1)
    # Only settings that lower compilation memory: the rest go to compile itself.
    with pytest.raises(TypeError, match="min_block_size"):
        low_memory(_Model, "w.safetensors", min_block_size=1)
    with pytest.raises(TypeError, match="weights"):
        low_memory(_Model)


def test_a_tensor_missing_from_the_file_is_named(tmp_path, caplog):
    # The safetensors package's save_file(state_dict) leaves out the non-persistent buffer.
    save_file(_Model().state_dict(), str(tmp_path / "no_buffer.safetensors"))
    with caplog.at_level(
        logging.WARNING, logger="torch_tensorrt.dynamo.tools.weight_mapping"
    ):
        with low_memory(_Model, tmp_path / "no_buffer.safetensors") as low_mem:
            assert low_mem.model.scale.is_meta
    assert "stay on the meta device" in caplog.text and "scale" in caplog.text


class _WithUnusedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 4)  # never called in forward

    def forward(self, x):
        return torch.relu(self.proj(x))


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_weight_the_graph_never_reads_may_stay_on_meta(tmp_path):
    torch.manual_seed(0)
    reference = _WithUnusedHead().eval()
    proj_only = {
        k: v for k, v in reference.state_dict().items() if k.startswith("proj.")
    }
    save_file(proj_only, str(tmp_path / "proj.safetensors"))
    with torch.device("meta"):
        model = _WithUnusedHead().eval()
    map_weights(model, tmp_path / "proj.safetensors", strict=False)
    assert model.head.weight.is_meta
    x = torch.randn(3, 8)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), inputs=[x.cuda()], min_block_size=1
    )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )


def test_leaving_drops_the_pages_also_on_an_exception(tmp_path, monkeypatch):
    reference = _weights(tmp_path / "w.safetensors")
    dropped = []

    def spy(storage):
        dropped.append(release_mapped_storage(storage))
        return dropped[-1]

    monkeypatch.setattr(_low_memory, "release_mapped_storage", spy)
    low_mem = low_memory(_Model, tmp_path / "w.safetensors")
    with pytest.raises(RuntimeError, match="boom"):
        with low_mem:
            assert not low_mem.model.proj.weight.is_meta
            raise RuntimeError("boom")
    assert dropped and None not in dropped  # every one a mapped weight
    assert low_mem.report.mapped_bytes > 0 and not low_mem.report.missing
    torch.testing.assert_close(low_mem.model.proj.weight, reference.proj.weight)


def test_a_block_that_keeps_weights_keeps_their_pages(tmp_path, monkeypatch):
    _weights(tmp_path / "w.safetensors")
    dropped = []
    monkeypatch.setattr(_low_memory, "release_mapped_storage", dropped.append)
    with low_memory(
        _Model, tmp_path / "w.safetensors", release_weights_after_build=False
    ):
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


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_with_the_block_settings(tmp_path, monkeypatch):
    reference = _weights(tmp_path / "w.safetensors")
    released = []
    real = torch_tensorrt.dynamo._weight_release.release_mapped_storage

    def spy(storage):
        released.append(real(storage))
        return released[-1]

    monkeypatch.setattr(
        torch_tensorrt.dynamo._weight_release, "release_mapped_storage", spy
    )
    x = torch.randn(3, 8)
    with low_memory(_Model, tmp_path / "w.safetensors") as low_mem:
        exported = torch.export.export(low_mem.model.eval(), (x,))
        trt_model = torch_tensorrt.compile(
            exported,
            ir="dynamo",
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )
    # By default, mapped weights' pages are dropped.
    assert released and None not in released
    torch.testing.assert_close(low_mem.model.proj.weight, reference.proj.weight)

    # With release_weights_after_build=False in the settings passed to compile, none are.
    released.clear()
    with low_memory(
        _Model, tmp_path / "w.safetensors", release_weights_after_build=False
    ) as low_mem:
        exported = torch.export.export(low_mem.model.eval(), (x,))
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_block_does_not_change_other_compiles(tmp_path):
    """A model compiled inside the block without its settings keeps its own: a GPU model isn't offloaded."""
    _weights(tmp_path / "w.safetensors")
    gpu_model = _Model().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    with low_memory(_Model, tmp_path / "w.safetensors"):
        torch_tensorrt.dynamo.compile(
            torch.export.export(gpu_model, (x,)),
            inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},
        )
    assert all(p.is_cuda for p in gpu_model.parameters())
    gpu_model(x)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_pytorch_part_reading_a_mapped_weight_runs(tmp_path):
    reference = _weights(tmp_path / "w.safetensors")
    x = torch.randn(3, 8)
    with low_memory(_Model, tmp_path / "w.safetensors") as low_mem:
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(low_mem.model.eval(), (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},  # reads the mapped scale
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
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
@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists("/proc/self/smaps"),
    reason="needs CUDA and /proc/self/smaps",
)
def test_the_weights_file_leaves_memory(tmp_path):
    path = os.path.realpath(tmp_path / "big.safetensors")
    save_weights_for_mapping(_Big().eval(), path)
    x = torch.randn(3, 512)
    with low_memory(_Big, path) as low_mem:
        low_mem.model(x)  # reads every weight in
        assert _resident_mib(path) >= 0.9
        torch_tensorrt.dynamo.compile(
            torch.export.export(low_mem.model.eval(), (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    # Only the pages at the weight's ends, which may hold a neighbour's data, can stay.
    assert _resident_mib(path) < 0.05
