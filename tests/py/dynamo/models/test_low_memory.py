# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import os

import pytest
import torch
import torch_tensorrt
from safetensors.torch import save_file
from torch_tensorrt.dynamo import _low_memory, low_memory
from torch_tensorrt.dynamo._low_memory import with_low_memory_defaults
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.tools import save_weights_for_mapping
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


@with_low_memory_defaults
def _settings(exported_program, **kwargs):
    return kwargs


def test_defaults_fill_in_what_the_caller_does_not_pass():
    assert _settings(None) == {}
    # A budget turns resource partitioning on.
    with low_memory(offload_module_to_cpu=True, cpu_memory_budget=3 << 30):
        assert _settings(None, cpu_memory_budget=5 << 30) == {
            "offload_module_to_cpu": True,
            "enable_resource_partitioning": True,
            "cpu_memory_budget": 5 << 30,
        }
        with low_memory(
            offload_module_to_cpu=False, enable_resource_partitioning=False
        ):
            assert _settings(None) == {
                "offload_module_to_cpu": False,
                "enable_resource_partitioning": False,
                "cpu_memory_budget": 3 << 30,
            }
        assert _settings(None) == {
            "offload_module_to_cpu": True,
            "enable_resource_partitioning": True,
            "cpu_memory_budget": 3 << 30,
        }
    assert _settings(None) == {}


def test_a_falsy_release_setting_releases_nothing():
    assert (
        CompilationSettings(release_weights_after_build=0).release_weights_after_build
        is False
    )
    with low_memory(release_weights_after_build=0):
        assert _settings(None) == {"release_weights_after_build": False}


def test_a_setting_left_at_none_is_not_set():
    with low_memory(release_weights_after_build=False):
        with low_memory(release_weights_after_build=None, offload_module_to_cpu=None):
            assert _settings(None) == {"release_weights_after_build": False}


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
        low_memory(optimisation_level=1)
    # Only settings that lower compilation memory: the rest go to compile itself.
    with pytest.raises(TypeError, match="min_block_size"):
        low_memory(min_block_size=1)
    with pytest.raises(TypeError, match="arg_inputs"):
        low_memory(arg_inputs=[])
    with pytest.raises(ValueError, match="together"):
        low_memory(_Model)


def test_a_tensor_left_on_meta_raises(tmp_path):
    # The safetensors package's save_file(state_dict) leaves out the non-persistent buffer.
    save_file(_Model().state_dict(), str(tmp_path / "no_buffer.safetensors"))
    with pytest.raises(ValueError, match="scale"):
        with low_memory(_Model, tmp_path / "no_buffer.safetensors"):
            pass
    assert _settings(None) == {}


def test_leaving_drops_the_pages_also_on_an_exception(tmp_path, monkeypatch):
    reference = _weights(tmp_path / "w.safetensors")
    dropped = []

    def spy(storage):
        dropped.append(release_mapped_storage(storage))
        return dropped[-1]

    monkeypatch.setattr(_low_memory, "release_mapped_storage", spy)
    ctx = low_memory(_Model, tmp_path / "w.safetensors", offload_module_to_cpu=True)
    with pytest.raises(RuntimeError, match="boom"):
        with ctx as model:
            assert not model.proj.weight.is_meta
            raise RuntimeError("boom")
    assert dropped and all(dropped)
    assert _settings(None) == {}
    assert ctx.report.mapped_bytes > 0 and not ctx.report.missing
    torch.testing.assert_close(model.proj.weight, reference.proj.weight)


def test_a_block_that_keeps_weights_keeps_their_pages(tmp_path, monkeypatch):
    _weights(tmp_path / "w.safetensors")
    dropped = []
    monkeypatch.setattr(_low_memory, "release_mapped_storage", dropped.append)
    with low_memory(
        _Model, tmp_path / "w.safetensors", release_weights_after_build=False
    ):
        pass
    assert not dropped


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_inside_the_block(tmp_path, monkeypatch):
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
    with low_memory(_Model, tmp_path / "w.safetensors") as model:
        exported = torch.export.export(model.eval(), (x,))
        trt_model = torch_tensorrt.compile(
            exported, ir="dynamo", inputs=[x.cuda()], min_block_size=1
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )
    assert released and all(released)  # by default, mapped weights' pages are dropped
    torch.testing.assert_close(model.proj.weight, reference.proj.weight)

    # The block's setting reaches compile: with release_weights_after_build=False, none are.
    released.clear()
    with low_memory(
        _Model, tmp_path / "w.safetensors", release_weights_after_build=False
    ) as model:
        exported = torch.export.export(model.eval(), (x,))
        torch_tensorrt.compile(
            exported, ir="dynamo", inputs=[x.cuda()], min_block_size=1
        )
    assert not released

    # An ordinary model's weights are left alone by default.
    plain = _Model().eval()
    exported = torch.export.export(plain, (x,))
    torch_tensorrt.dynamo.compile(exported, inputs=[x.cuda()], min_block_size=1)
    assert plain.proj.weight.untyped_storage().nbytes() > 0


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_pytorch_part_reading_a_mapped_weight_runs(tmp_path):
    reference = _weights(tmp_path / "w.safetensors")
    x = torch.randn(3, 8)
    with low_memory(_Model, tmp_path / "w.safetensors") as model:
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model.eval(), (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},  # reads the mapped scale
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
    with low_memory(_Big, path) as model:
        model(x)  # reads every weight in
        assert _resident_mib(path) >= 0.9
        torch_tensorrt.dynamo.compile(
            torch.export.export(model.eval(), (x,)), inputs=[x.cuda()], min_block_size=1
        )
    # Only the pages at the weight's ends, which may hold a neighbour's data, can stay.
    assert _resident_mib(path) < 0.05
