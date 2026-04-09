"""
Tests for SymInt scalar input handling in symbolic shape capture and TRT compilation.

These tests verify that when Dynamo partitions an FX graph such that a SymInt
(e.g., from targets.size(0)) becomes a bare scalar placeholder input to the TRT
subgraph, the symbolic shape extraction and compilation succeed.

This covers the fix in _symbolic_shape_capture.py where non-tensor inputs
(SymInt, int, float, bool) are handled gracefully instead of aborting extraction.
"""

import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


@pytest.mark.unit
@pytest.mark.parametrize("use_python_runtime", [True, False])
def test_symint_from_size_used_in_reshape(use_python_runtime):
    """
    Test that a SymInt derived from tensor.size(0) can be used in reshape
    when it becomes a scalar placeholder input to the TRT subgraph.

    This is the core pattern from issue #4107: targets.size(0) produces a
    SymInt that Dynamo passes as a bare scalar input to the TRT partition,
    which then uses it in a reshape operation.
    """

    class Model(torch.nn.Module):
        def forward(self, x, targets):
            B = targets.size(0)
            y = x.reshape(B, -1)
            return y

    model = Model().eval().cuda()

    x = torch.randn(16, 64).cuda()
    targets = torch.randint(0, 10, (16, 1), dtype=torch.int64).cuda()

    torch._dynamo.mark_dynamic(x, 0, min=1, max=2048)
    torch._dynamo.mark_dynamic(targets, 0, min=1, max=2048)

    compile_spec = {
        "min_block_size": 1,
        "pass_through_build_failures": True,
        "use_python_runtime": use_python_runtime,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    output_ref = model(x, targets)
    output_trt = trt_model(x, targets)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"SymInt reshape test (python_runtime={use_python_runtime}) failed. Cosine sim: {cos_sim}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("use_python_runtime", [True, False])
def test_scalar_tensor_input(use_python_runtime):
    """
    Test that a 0-dim scalar tensor input (e.g., cache_length) is handled
    correctly during symbolic shape extraction and TRT compilation.
    """

    class Model(torch.nn.Module):
        def forward(self, x, offset):
            return x + offset

    model = Model().eval().cuda()

    x = torch.randn(16, 64).cuda()
    offset = torch.tensor(5.0).cuda()

    compile_spec = {
        "min_block_size": 1,
        "pass_through_build_failures": True,
        "use_python_runtime": use_python_runtime,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    output_ref = model(x, offset)
    output_trt = trt_model(x, offset)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Scalar tensor input test (python_runtime={use_python_runtime}) failed. Cosine sim: {cos_sim}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("use_python_runtime", [True, False])
def test_symint_with_index_and_reshape(use_python_runtime):
    """
    Full reproduction of issue #4107 pattern: symbolic size from int64 tensor,
    used with index operation and reshape.

    Model does:
    1. B = targets.size(0)  → SymInt
    2. idx = cache_length + arange(1) → int64 index tensor
    3. y = x[:, idx, :] → gather with int64 index
    4. z = y.reshape(B, 1, -1, 2) → reshape using SymInt
    """

    class TestModule(torch.nn.Module):
        def forward(self, x, targets, cache_length):
            B = targets.size(0)
            idx = cache_length + torch.arange(1, device=x.device)
            y = x[:, idx, :]
            z = y.reshape(B, 1, -1, 2)
            return z

    model = TestModule().eval().cuda()

    B, S, D = 16, 128, 1024
    x = torch.randn(B, S, D).cuda()
    targets = torch.randint(0, 10, (B, 1), dtype=torch.int64).cuda()
    cache_length = torch.tensor(0, dtype=torch.int64).cuda()

    torch._dynamo.mark_dynamic(targets, 0, min=1, max=2048)
    torch._dynamo.mark_dynamic(x, 0, min=1, max=2048)

    compile_spec = {
        "min_block_size": 1,
        "truncate_double": True,
        "pass_through_build_failures": True,
        "use_python_runtime": use_python_runtime,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    output_ref = model(x, targets, cache_length)
    output_trt = trt_model(x, targets, cache_length)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Issue 4107 repro test (python_runtime={use_python_runtime}) failed. Cosine sim: {cos_sim}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("use_python_runtime", [True, False])
def test_symint_with_different_batch_sizes(use_python_runtime):
    """
    Test that after compilation with a SymInt scalar input, the model
    produces correct results with different batch sizes.
    """

    class Model(torch.nn.Module):
        def forward(self, x, targets):
            B = targets.size(0)
            return x.reshape(B, 2, -1)

    model = Model().eval().cuda()

    x = torch.randn(8, 64).cuda()
    targets = torch.randint(0, 10, (8, 1), dtype=torch.int64).cuda()

    torch._dynamo.mark_dynamic(x, 0, min=1, max=2048)
    torch._dynamo.mark_dynamic(targets, 0, min=1, max=2048)

    compile_spec = {
        "min_block_size": 1,
        "pass_through_build_failures": True,
        "use_python_runtime": use_python_runtime,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    for batch_size in [4, 8, 16]:
        x_test = torch.randn(batch_size, 64).cuda()
        targets_test = torch.randint(0, 10, (batch_size, 1), dtype=torch.int64).cuda()

        output_ref = model(x_test, targets_test)
        output_trt = trt_model(x_test, targets_test)

        cos_sim = cosine_similarity(output_ref, output_trt)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Varying batch size test (python_runtime={use_python_runtime}) failed at B={batch_size}. Cosine sim: {cos_sim}",
        )

    torch._dynamo.reset()
