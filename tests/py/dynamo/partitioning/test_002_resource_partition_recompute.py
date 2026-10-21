# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt


class _MaskedLayers(torch.nn.Module):
    """Attention-like layers that all read one mask built from the inputs."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(256, 2048, bias=False).half() for _ in range(8)
        )

    def forward(self, x, pad):
        mask = torch.where(pad[:, None, :] & pad[:, :, None], 0.0, -1e4).half()
        for layer in self.layers:
            x = (
                torch.softmax(x @ x.transpose(1, 2) / 16 + mask, -1)
                @ layer(x)[..., :256]
            )
        return x


def test_engines_recompute_the_input_only_mask():
    """Each engine builds the mask from the padding input itself, which only one engine
    reads, instead of receiving the 512x512 mask (512 KiB) from the first engine."""
    torch.manual_seed(0)
    model = _MaskedLayers().cuda().eval()
    x = torch.randn(1, 512, 256, device="cuda", dtype=torch.half) / 8
    pad = torch.ones(1, 512, dtype=torch.bool, device="cuda")
    pad[:, 400:] = False
    trt = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x, pad)),
        arg_inputs=(x, pad),
        min_block_size=1,
        enable_resource_partitioning=True,
        resource_partition_engine_cap=3 << 19,  # one 1 MiB layer per engine
    )
    engines = [m for n, m in trt.named_children() if "resource_split" in n]
    masks, pad_readers = [], []
    for engine in engines:
        engine.register_forward_pre_hook(
            lambda _, args: masks.extend(
                a
                for a in args
                if isinstance(a, torch.Tensor) and a.shape[-2:] == (512, 512)
            )
        )
        engine.register_forward_pre_hook(
            lambda m, args: pad_readers.extend(
                [m] if any(a is pad for a in args) else []
            )
        )
    with torch.no_grad():
        torch.testing.assert_close(trt(x, pad), model(x, pad), rtol=1e-2, atol=1e-2)
    assert len(engines) >= 4
    assert masks == []
    assert len(pad_readers) <= 1
