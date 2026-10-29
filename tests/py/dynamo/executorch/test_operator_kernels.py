# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from torch_tensorrt.executorch._export_utils import (  # noqa: E402
    operators_without_kernels,
)


def _program(*operators):
    plan = SimpleNamespace(
        operators=[SimpleNamespace(name=name, overload=o) for name, o in operators]
    )
    return SimpleNamespace(executorch_program=SimpleNamespace(execution_plan=[plan]))


@pytest.mark.unit
def test_operators_without_kernels_names_the_ones_the_runtime_lacks(monkeypatch):
    portable_lib = pytest.importorskip("executorch.extension.pybindings.portable_lib")
    monkeypatch.setattr(
        portable_lib,
        "_get_operator_names",
        lambda: ["aten::add.out", "aten::_local_scalar_dense"],
    )
    program = _program(
        ("aten::add", "out"),
        # The registry names an operator without an overload by its name alone.
        ("aten::_local_scalar_dense", ""),
        ("aten::multinomial", "out"),
    )
    assert operators_without_kernels(program) == ["aten::multinomial.out"]


@pytest.mark.unit
def test_save_warns_about_an_operator_the_runtime_lacks(tmp_path, caplog):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + TensorRT for a real engine")
    pytest.importorskip("executorch.extension.pybindings.portable_lib")

    class Resize(torch.nn.Module):
        # An antialiased bicubic resize has no Torch-TensorRT converter, so it stays
        # in PyTorch, and ExecuTorch has no kernel for it either.
        def forward(self, x):
            resized = torch.nn.functional.interpolate(
                x * 2.0, size=(8, 8), mode="bicubic", antialias=True
            )
            return resized + 1.0

    model = Resize().eval().cuda()
    x = torch.rand(1, 3, 16, 16, device="cuda")
    trt_module = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), arg_inputs=[x], min_block_size=1
    )
    pte = tmp_path / "model.pte"
    with caplog.at_level(logging.WARNING):
        torch_tensorrt.save(
            trt_module,
            str(pte),
            output_format="executorch",
            arg_inputs=[x],
            retrace=False,
        )
    assert pte.exists()
    assert "aten::_upsample_bicubic2d_aa.out" in caplog.text
