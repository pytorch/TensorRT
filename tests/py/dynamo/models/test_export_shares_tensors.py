# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from torch_tensorrt.dynamo._exporter import export


class _WithPyTorchLayers(torch.nn.Module):
    """A parameter and a buffer read by ops that run in PyTorch, so both stay tensors of the compiled module."""

    def __init__(self, size=64):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(size), requires_grad=False)
        self.register_buffer("shift", torch.randn(size))

    def forward(self, x):
        return torch.relu(x) * self.scale - self.shift


def _compile(model, x):
    return torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)),
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.mul.Tensor, torch.ops.aten.sub.Tensor},
    )


def test_export_copies_the_compiled_modules_tensors_unless_asked_to_share():
    torch.manual_seed(0)
    model = _WithPyTorchLayers().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    with torch.no_grad():
        expected = model(x).clone()
    before = {name: t.detach().clone() for name, t in model.state_dict().items()}
    trt = _compile(model, x)
    for use_legacy_exporter in (True, False):
        for share_tensors in (False, True):
            ep = export(
                trt,
                arg_inputs=[x],
                use_legacy_exporter=use_legacy_exporter,
                share_tensors=share_tensors,
            )
            exported = {
                t.untyped_storage().data_ptr()
                for t in [*ep.state_dict.values(), *ep.constants.values()]
                if isinstance(t, torch.Tensor)
            }
            # A copy keeps a program exported before an in-place refit consistent with its engines.
            assert (
                model.scale.untyped_storage().data_ptr() in exported
            ) == share_tensors
            assert (
                model.shift.untyped_storage().data_ptr() in exported
            ) == share_tensors
            with torch.no_grad():
                torch.testing.assert_close(
                    ep.module()(x), expected, rtol=1e-2, atol=1e-2
                )

    # Exporting left the shared tensors and the compiled module as they were.
    for name, t in model.state_dict().items():
        torch.testing.assert_close(t, before[name], rtol=0, atol=0)
    assert any("_run_on_acc" in name for name, _ in trt.named_children())
    with torch.no_grad():
        torch.testing.assert_close(trt(x), expected, rtol=1e-2, atol=1e-2)


def test_save_does_not_copy_the_compiled_modules_weights(tmp_path):
    size = 1 << 24
    model = _WithPyTorchLayers(size).eval().cuda()
    x = torch.randn(size, device="cuda")
    trt = _compile(model, x)
    weight_bytes = model.scale.nbytes + model.shift.nbytes
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch_tensorrt.save(trt, str(tmp_path / "model.ep"), arg_inputs=[x], retrace=False)
    # A copy of the module would add weight_bytes on the GPU.
    assert torch.cuda.max_memory_allocated() - before < weight_bytes / 2
