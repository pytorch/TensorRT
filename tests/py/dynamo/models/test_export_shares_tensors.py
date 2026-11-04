# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from torch_tensorrt.dynamo._exporter import export


class _WithPyTorchLayers(torch.nn.Module):
    """A parameter and a buffer read by ops that run in PyTorch, so both stay tensors of the compiled module."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 64)
        self.scale = torch.nn.Parameter(torch.randn(64), requires_grad=False)
        self.register_buffer("shift", torch.randn(64))

    def forward(self, x):
        return self.a(x) * self.scale - self.shift


def test_export_shares_the_compiled_modules_tensors():
    torch.manual_seed(0)
    model = _WithPyTorchLayers().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    with torch.no_grad():
        expected = model(x).clone()
    before = {name: t.detach().clone() for name, t in model.state_dict().items()}
    trt = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.mul.Tensor, torch.ops.aten.sub.Tensor},
    )
    for use_legacy_exporter in (True, False):
        ep = export(trt, arg_inputs=[x], use_legacy_exporter=use_legacy_exporter)
        exported = {
            t.untyped_storage().data_ptr()
            for t in [*ep.state_dict.values(), *ep.constants.values()]
            if isinstance(t, torch.Tensor)
        }
        # The exported program references the parameter and the buffer themselves; they aren't copied.
        assert model.scale.untyped_storage().data_ptr() in exported
        assert model.shift.untyped_storage().data_ptr() in exported
        with torch.no_grad():
            torch.testing.assert_close(ep.module()(x), expected, rtol=1e-2, atol=1e-2)

    # Exporting left the shared tensors and the compiled module as they were.
    for name, t in model.state_dict().items():
        torch.testing.assert_close(t, before[name], rtol=0, atol=0)
    assert any("_run_on_acc" in name for name, _ in trt.named_children())
    with torch.no_grad():
        torch.testing.assert_close(trt(x), expected, rtol=1e-2, atol=1e-2)
