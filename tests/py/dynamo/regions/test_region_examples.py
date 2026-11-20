# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep public region examples executable; API edge cases live separately.

The last test pins a model the tutorial cannot promise yet: it is expected to
fail until the product is fixed.
"""

from pathlib import Path

import pytest
import torch
import torch_tensorrt
from torch.utils._triton import has_triton
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.region import ExecuteInTorch

from ._utils import run_python

ROOT = Path(__file__).resolve().parents[4]
# Each example and the success message it prints.
EXAMPLES = {
    "execute_in_torch.py": "Both eager and inductor regions match PyTorch",
    "fused_op.py": "The supplied custom operator runs in PyTorch",
    "fused_kernel.py": "One TensorRT engine includes the fused kernel",
    "region_rules.py": "Rules placed the same regions on all three compile paths",
}
# The examples that import Triton or compile a region with CUDA Inductor.
NEEDS_TRITON = ("execute_in_torch.py", "fused_kernel.py")


@pytest.mark.parametrize("script", EXAMPLES)
def test_region_example_uses_only_public_api(script):
    # Users copy examples; internal modules, metadata keys and attributes can change.
    source = (ROOT / "examples" / "dynamo" / script).read_text(encoding="utf-8")
    for internal in ("region_compilation", "torch_tensorrt_region", "_torch_tensorrt_"):
        assert internal not in source, f"{script} uses internal {internal!r}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("script", "success"), EXAMPLES.items())
def test_region_example(script, success):
    if script in NEEDS_TRITON:
        pytest.importorskip("triton")
    if script == "execute_in_torch.py" and not has_triton():
        pytest.skip("CUDA Inductor requires Triton support on the device")
    if script == "fused_kernel.py" and not torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        pytest.skip("TensorRT QDP plugin support is required")
    output = run_python(ROOT / "examples" / "dynamo" / script, cwd=ROOT, timeout=180)
    assert success in output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source", ["rule", "scope"])
def test_forward_argument_named_input_compiles_with_regions(source):
    # Copying a program's graph renames a node called like a builtin.
    class Scoped(torch.nn.Module):
        def forward(self, input):
            with torch_tensorrt.region.execute_in_torch():
                result = torch.relu(input + 1)
            return result * 2

    x = torch.randn(4, 16, device="cuda")
    if source == "rule":
        # torch.nn.Sequential.forward names its argument "input".
        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
        rules = {"1": ExecuteInTorch()}
    else:
        model, rules = Scoped(), None
    model = model.eval().cuda()
    compiled = torch_tensorrt.compile(
        model, inputs=[x], ir="dynamo", min_block_size=1, regions=rules
    )
    assert len(get_region_records(compiled)) == 1
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-2, atol=1e-2)
