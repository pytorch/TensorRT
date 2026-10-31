# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch_tensorrt
from safetensors.torch import save_file
from torch_tensorrt.dynamo import _low_memory, low_memory
from torch_tensorrt.dynamo._low_memory import with_low_memory_defaults
from torch_tensorrt.dynamo.tools import release_mapped_storage, save_weights_for_mapping


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
    with low_memory(optimization_level=1, min_block_size=3):
        assert _settings(None, min_block_size=5) == {
            "optimization_level": 1,
            "min_block_size": 5,
        }
        with low_memory(optimization_level=2):
            assert _settings(None) == {"optimization_level": 2, "min_block_size": 3}
        assert _settings(None) == {"optimization_level": 1, "min_block_size": 3}
    assert _settings(None) == {}


def test_bad_arguments_raise():
    with pytest.raises(TypeError, match="optimisation_level"):
        low_memory(optimisation_level=1)
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
    ctx = low_memory(_Model, tmp_path / "w.safetensors", optimization_level=1)
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
    with low_memory(_Model, tmp_path / "w.safetensors", min_block_size=1) as model:
        exported = torch.export.export(model.eval(), (x,))
        trt_model = torch_tensorrt.compile(exported, ir="dynamo", inputs=[x.cuda()])
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )
    assert any(  # the block's min_block_size=1: one op is enough for an engine
        "_run_on_acc" in name for name, _ in trt_model.named_children()
    )
    assert released and all(released)  # by default, mapped weights' pages are dropped
    torch.testing.assert_close(model.proj.weight, reference.proj.weight)

    # An ordinary model's weights are left alone by default.
    plain = _Model().eval()
    exported = torch.export.export(plain, (x,))
    torch_tensorrt.dynamo.compile(exported, inputs=[x.cuda()], min_block_size=1)
    assert plain.proj.weight.untyped_storage().nbytes() > 0
