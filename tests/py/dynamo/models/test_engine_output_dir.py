# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo._settings import CompilationSettings


def test_engine_output_dir_default_and_old_pickle():
    assert CompilationSettings().engine_output_dir is None
    state = CompilationSettings().__dict__.copy()
    state.pop("engine_output_dir")
    restored = CompilationSettings.__new__(CompilationSettings)
    restored.__setstate__(state)
    assert restored.engine_output_dir is None


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


def _compile(tmp_path, **kwargs):
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


def test_engines_stay_on_disk_until_first_use(tmp_path):
    model, x, trt, engines = _compile(tmp_path, engine_output_dir=str(tmp_path))
    assert len(engines) == 2
    files = sorted(glob.glob(str(tmp_path / "*.engine")))
    assert sorted(m.serialized_engine_path for m in engines) == files
    # Nothing loaded on the GPU, nothing held on the host.
    assert all(m.engine is None and m.serialized_engine is None for m in engines)
    # The last engine's outputs are marked unowned at compile time and applied when it loads.
    assert engines[-1].are_output_tensors_unowned()

    with torch.no_grad():
        torch.testing.assert_close(trt(x), model(x), rtol=1e-2, atol=1e-2)
    assert all(m.engine is not None for m in engines)
    assert engines[-1].engine.are_output_tensors_unowned()


def test_exported_program_save_loads_the_engines(tmp_path):
    model, x, trt, engines = _compile(tmp_path, engine_output_dir=str(tmp_path))
    path = str(tmp_path / "trt.ep")
    torch_tensorrt.save(trt, path, output_format="exported_program", arg_inputs=[x])
    loaded = torch.export.load(path).module()
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x), rtol=1e-2, atol=1e-2)


def test_executorch_save_reads_engines_from_their_files(tmp_path, monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    _, x, trt, engines = _compile(tmp_path, engine_output_dir=str(tmp_path / "engines"))
    setups = []
    real_setup = TorchTensorRTModule.setup_engine

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", counting_setup)
    pte_path = str(tmp_path / "model.pte")
    torch_tensorrt.save(
        trt, pte_path, output_format="executorch", arg_inputs=[x], retrace=False
    )
    # The save copied each engine from its file without loading any (the exporter works on a copy
    # of the module, so count setups rather than look at the originals).
    assert setups == []
    pte = open(pte_path, "rb").read()
    for m in engines:
        assert pte.find(open(m.serialized_engine_path, "rb").read()) >= 0

    # Same program as a save from engines in memory: the engines are byte-identical builds here,
    # so only the size of the pickled settings (which carry engine_output_dir) differs.
    _, _, trt_mem, _ = _compile(tmp_path)
    mem_path = str(tmp_path / "memory.pte")
    torch_tensorrt.save(
        trt_mem, mem_path, output_format="executorch", arg_inputs=[x], retrace=False
    )
    mem = open(mem_path, "rb").read()
    for m in engines:
        assert mem.find(open(m.serialized_engine_path, "rb").read()) >= 0
    assert abs(len(pte) - len(mem)) < 1024


@pytest.mark.skipif(not os.path.exists("/proc/self/maps"), reason="reads /proc")
def test_executorch_backend_drops_only_engine_file_pages(tmp_path):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.executorch.backend import _drop_engine_file_pages

    def rss_file_mib():
        status = dict(line.split(":", 1) for line in open("/proc/self/status"))
        return int(status["RssFile"].split()[0]) / 1024

    data = os.urandom(64 << 20)
    path = tmp_path / "x.engine"
    path.write_bytes(data)
    engine = torch.from_file(str(path), shared=False, size=len(data), dtype=torch.uint8)
    assert engine.numpy().tobytes() == data
    before = rss_file_mib()
    _drop_engine_file_pages(engine)
    assert before - rss_file_mib() > 60
    assert engine.numpy().tobytes() == data

    anonymous = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    _drop_engine_file_pages(anonymous)
    assert anonymous.numpy().tobytes() == data


def test_engine_files_are_not_needed_in_memory_mode(tmp_path):
    _, _, trt, engines = _compile(tmp_path)
    assert all(m.serialized_engine_path is None for m in engines)
    assert all(m.engine is not None for m in engines)
    assert not glob.glob(os.path.join(str(tmp_path), "*.engine"))
