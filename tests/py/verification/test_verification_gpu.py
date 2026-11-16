# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Real TensorRT integration; runs only with a CUDA-capable installation."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_artifact_validation_and_cuda_timing(tmp_path):
    import torch_tensorrt
    from torch_tensorrt.verification import InputSample, benchmark, validate

    class Add(torch.nn.Module):
        def forward(self, x):
            return x + x

    reference = Add().cuda().eval()
    tensor = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    exported = torch.export.export(reference, (tensor,))
    compiled = torch_tensorrt.dynamo.compile(
        exported, inputs=[tensor], enabled_precisions={torch.float16}, min_block_size=1
    )
    path = tmp_path / "compiled.pt2"
    torch_tensorrt.save(compiled, str(path), arg_inputs=[tensor])
    loaded = torch_tensorrt.load(str(path)).module()
    sample = InputSample(args=(tensor,))
    assert validate(reference, loaded, [sample], atol=1e-3, rtol=1e-3).passed
    result = benchmark(reference, loaded, sample, warmup=2, iterations=5)
    assert len(result.compiled.device_samples_ms) == 5
    assert (
        result.speedup["device"]
        == result.reference.device_statistics["median"]
        / result.compiled.device_statistics["median"]
    )
    assert (
        result.speedup["wall"]
        == result.reference.wall_statistics["median"]
        / result.compiled.wall_statistics["median"]
    )
    assert result.to_json()
