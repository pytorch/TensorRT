# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import tempfile

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402
from torch_tensorrt.executorch import _compile_and_save  # noqa: E402


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


@pytest.mark.unit
def test_engines_stream_from_their_files_into_the_program(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    setups, streamed = [], {}
    real_setup = TorchTensorRTModule.setup_engine
    real_add = NamedDataStore.add_named_data

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                streamed[data._path] = f.read()
        else:
            streamed[key] = data
        return real_add(self, key, data, *args, **kwargs)

    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", counting_setup)
    monkeypatch.setattr(NamedDataStore, "add_named_data", recording_add)

    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.executorch.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )

    # No engine was ever set up, and each one reached the program as its own file.
    assert setups == []
    assert len(streamed) == 2
    assert all(path.endswith(".engine") for path in streamed)
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    # The files are gone once the program is saved.
    assert not glob.glob(str(tmp_path / "*.engine"))


def test_keywords_go_to_compile_or_save_by_name(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        _compile_and_save,
        "dynamo_compile",
        lambda program, **kwargs: calls.setdefault("compile", kwargs) and "trt_gm",
    )
    monkeypatch.setattr(
        _compile_and_save,
        "save",
        lambda module, path, **kwargs: calls.setdefault("save", kwargs),
    )
    _compile_and_save.compile_and_save(
        "program",
        "model.pte",
        arg_inputs=["x"],
        min_block_size=1,
        external_engine_data="model",
    )
    assert calls["compile"] == {
        "offload_compiled_engines_to_disk": True,
        "arg_inputs": ["x"],
        "min_block_size": 1,
    }
    assert calls["save"] == {
        "output_format": "executorch",
        "arg_inputs": ["x"],
        "external_engine_data": "model",
    }

    with pytest.raises(TypeError, match="min_blok_size"):
        _compile_and_save.compile_and_save("program", "model.pte", min_blok_size=1)
    with pytest.raises(ValueError, match="always offloads"):
        _compile_and_save.compile_and_save(
            "program", "model.pte", offload_compiled_engines_to_disk=False
        )
