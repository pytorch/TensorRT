# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Layout regressions for the cuTile frontend example."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

import torch_tensorrt.kernels as ttk

from .conftest import skip_no_cuda, skip_no_cutile


def _load_example(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "examples/dynamo/cutile_op.py"
    spec = importlib.util.spec_from_file_location("_ttk_cutile_example", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(ttk, "cutile_op", lambda *args, **kwargs: None)
    spec.loader.exec_module(module)
    return module


def _check_transposed_input(example, device):
    x = torch.arange(15, dtype=torch.float32, device=device).reshape(3, 5).t()
    assert not x.is_contiguous()

    out = example.add_one_eager(x)
    meta = example.add_one_meta(x)
    assert out.is_contiguous()
    assert meta.stride() == out.stride()
    assert meta.shape == out.shape == x.shape
    assert meta.dtype == out.dtype == x.dtype
    torch.testing.assert_close(out, x + 1)


def test_example_eager_writes_returned_output_for_transposed_input(monkeypatch):
    """Exercise the allocation and flattening without a GPU or cuTile install."""
    cuda = ModuleType("cuda")
    ct = ModuleType("cuda.tile")
    cuda.tile = ct
    ct.kernel = lambda kernel: kernel
    ct.Constant = list  # Only used as a subscriptable annotation in the example.
    ct.cdiv = lambda value, divisor: (value + divisor - 1) // divisor

    def launch(stream, grid, kernel, args):
        flat_x, flat_out, tile_size = args
        flat_out.copy_(flat_x + 1)

    ct.launch = launch
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.tile", ct)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    _check_transposed_input(_load_example(monkeypatch), "cpu")


@skip_no_cuda
@skip_no_cutile
def test_example_eager_transposed_input_on_cuda(monkeypatch):
    _check_transposed_input(_load_example(monkeypatch), "cuda")
