# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import glob
import tempfile

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import _compiler
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import _conversion
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


def test_default_and_old_pickle():
    assert CompilationSettings().offload_compiled_engines_to_disk is False
    state = CompilationSettings().__dict__.copy()
    state.pop("offload_compiled_engines_to_disk")
    restored = CompilationSettings.__new__(CompilationSettings)
    restored.__setstate__(state)
    assert restored.offload_compiled_engines_to_disk is False


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


def _compile(**kwargs):
    torch.manual_seed(0)
    model = _TwoEngines().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    trt = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
        **kwargs,
    )
    engines = [m for n, m in trt.named_children() if "_run_on_acc" in n]
    return model, x, trt, engines


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_engines_wait_on_disk_until_the_last_is_built(temp_dir, monkeypatch):
    events = []
    real_offload = _conversion.offload_engine
    real_setup = TorchTensorRTModule.setup_engine

    def recording_offload(*args, **kwargs):
        events.append("offloaded")
        return real_offload(*args, **kwargs)

    def recording_setup(self):
        if self.engine is None:
            events.append("set up")
        real_setup(self)

    monkeypatch.setattr(_conversion, "offload_engine", recording_offload)
    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", recording_setup)
    model, x, trt, engines = _compile(offload_compiled_engines_to_disk=True)

    # No engine is set up before the last one is built, and then none keeps its mapped plan.
    assert events == ["offloaded", "offloaded", "set up", "set up"]
    assert all(m.engine is not None for m in engines)
    assert not any(isinstance(m._serialized_engine, torch.Tensor) for m in engines)
    assert engines[-1].are_output_tensors_unowned()
    assert not glob.glob(str(temp_dir / "*.engine"))
    with torch.no_grad():
        torch.testing.assert_close(trt(x), model(x), rtol=1e-2, atol=1e-2)


def test_engine_files_are_deleted_when_compilation_fails(temp_dir, monkeypatch):
    real_convert = _compiler.convert_module
    calls = []

    def failing_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("build failed")
        return real_convert(*args, **kwargs)

    monkeypatch.setattr(_compiler, "convert_module", failing_second)
    with pytest.raises(RuntimeError, match="build failed"):
        _compile(offload_compiled_engines_to_disk=True)
    gc.collect()
    assert len(calls) == 2
    assert not glob.glob(str(temp_dir / "*.engine"))


def test_without_the_setting_no_engine_goes_to_disk(temp_dir):
    _, _, _, engines = _compile()
    assert all(m.engine is not None for m in engines)
    assert not glob.glob(str(temp_dir / "*.engine"))
