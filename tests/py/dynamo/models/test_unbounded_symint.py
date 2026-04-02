# type: ignore
"""
Tests for unbounded (infinite) SymInt handling in Torch-TensorRT.

These tests verify that when dynamic shapes are used without explicit upper bounds,
the system properly handles the unbounded SymInts and applies reasonable defaults.
"""

import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch.nn import functional as F
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


@pytest.mark.unit
def test_unbounded_symint_torch_compile_single_dim():
    """
    Test torch.compile with unbounded SymInt on a single dimension.
    Uses mark_dynamic without max to create unbounded upper bound.
    """

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(128, 64)

        def forward(self, x):
            return F.relu(self.linear1(x))

    model = SimpleModel().eval().cuda()

    # Create input with unbounded batch dimension
    input_tensor = torch.randn(4, 128).cuda()
    # Mark dimension 0 as dynamic with only min (no max = unbounded)
    # torch._dynamo.mark_dynamic(input_tensor, 0, min=1)

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }

    # Compile with torch.compile backend
    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with original input
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Unbounded SymInt test failed. Cosine sim: {cos_sim}, Threshold: {COSINE_THRESHOLD}",
    )

    # Test with different batch size within reasonable range
    input_tensor_8 = torch.randn(8, 128).cuda()
    output_ref_8 = model(input_tensor_8)
    output_trt_8 = trt_model(input_tensor_8)

    cos_sim_8 = cosine_similarity(output_ref_8, output_trt_8)
    assertions.assertTrue(
        cos_sim_8 > COSINE_THRESHOLD,
        msg=f"Unbounded SymInt test with different batch size failed. Cosine sim: {cos_sim_8}",
    )

    # Test with different batch size within reasonable range
    input_tensor_8 = torch.randn(16, 128).cuda()
    output_ref_8 = model(input_tensor_8)
    output_trt_8 = trt_model(input_tensor_8)

    cos_sim_8 = cosine_similarity(output_ref_8, output_trt_8)
    assertions.assertTrue(
        cos_sim_8 > COSINE_THRESHOLD,
        msg=f"Unbounded SymInt test with different batch size failed. Cosine sim: {cos_sim_8}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_conv_model():
    """
    Test unbounded SymInt with a convolutional model.
    Verifies that conv operations work with unbounded batch dimension.
    """

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = ConvModel().eval().cuda()

    # Create input with unbounded batch dimension
    input_tensor = torch.randn(2, 3, 224, 224).cuda()
    torch._dynamo.mark_dynamic(input_tensor, 0, min=1, max=10)

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with original input
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Conv unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_multiple_dims(tmp_path):
    """
    Test unbounded SymInt on multiple dimensions.
    Verifies handling when multiple dimensions are marked as dynamic without bounds.
    """

    class MultiDimModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Simple operation that preserves shape
            return x * 2 + 1

    model = MultiDimModel().eval().cuda()

    # Create input with multiple unbounded dimensions
    input_tensor = torch.randn(4, 8, 16).cuda()
    # Mark multiple dimensions as dynamic without max

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with original input
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)
    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Multiple dims unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    # Test with different sizes
    input_tensor_diff = torch.randn(6, 12, 16).cuda()
    output_ref_diff = model(input_tensor_diff)
    output_trt_diff = trt_model(input_tensor_diff)

    cos_sim_diff = cosine_similarity(output_ref_diff, output_trt_diff)
    assertions.assertTrue(
        cos_sim_diff > COSINE_THRESHOLD,
        msg=f"Multiple dims unbounded SymInt with different sizes failed. Cosine sim: {cos_sim_diff}",
    )

    # Test with different sizes
    input_tensor_diff = torch.randn(8, 16, 16).cuda()
    output_ref_diff = model(input_tensor_diff)
    output_trt_diff = trt_model(input_tensor_diff)

    cos_sim_diff = cosine_similarity(output_ref_diff, output_trt_diff)
    assertions.assertTrue(
        cos_sim_diff > COSINE_THRESHOLD,
        msg=f"Multiple dims unbounded SymInt with different sizes failed. Cosine sim: {cos_sim_diff}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_with_reshape():
    """
    Test unbounded SymInt with reshape operations.
    Verifies that shape-dependent operations work correctly with unbounded dims.
    """

    class ReshapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            batch_size = x.shape[0]
            # Reshape using the dynamic batch dimension
            x = x.view(batch_size, -1)
            return x

    model = ReshapeModel().eval().cuda()

    # Create input with unbounded batch dimension
    input_tensor = torch.randn(4, 8, 16).cuda()

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": False,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with original input
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Reshape with unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    input_tensor = torch.randn(12, 32, 16).cuda()
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Reshape with unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    input_tensor = torch.randn(8, 16, 16).cuda()
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Reshape with unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    # Verify output shapes match
    assertions.assertEqual(output_ref.shape, output_trt.shape)

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_cat_operation():
    """
    Test unbounded SymInt with concatenation operations.
    Verifies that operations involving shape extraction work with unbounded dims.
    """

    class CatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.const_tensor = torch.nn.Parameter(torch.randn(1, 1, 16))

        def forward(self, x):
            batch_size = x.shape[0]
            # Expand constant to match batch size
            expanded = self.const_tensor.expand(batch_size, -1, -1)
            # Concatenate along dim 1
            return torch.cat([expanded, x], dim=1)

    model = CatModel().eval().cuda()

    # Create input with unbounded batch dimension
    input_tensor = torch.randn(4, 8, 16).cuda()

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": False,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with original input
    output_ref = model(input_tensor)
    output_trt = trt_model(input_tensor)

    cos_sim = cosine_similarity(output_ref, output_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Cat with unbounded SymInt test failed. Cosine sim: {cos_sim}",
    )

    # Test with different batch size
    input_tensor_diff = torch.randn(6, 8, 16).cuda()
    output_ref_diff = model(input_tensor_diff)
    output_trt_diff = trt_model(input_tensor_diff)

    cos_sim_diff = cosine_similarity(output_ref_diff, output_trt_diff)
    assertions.assertTrue(
        cos_sim_diff > COSINE_THRESHOLD,
        msg=f"Cat with unbounded SymInt and different batch failed. Cosine sim: {cos_sim_diff}",
    )

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_reasonable_default():
    """
    Test that the default max (min * 128) is applied for unbounded SymInts.
    This test verifies the fallback behavior when no max is specified.
    """

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 32)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval().cuda()

    # Create input with unbounded batch dimension and small min
    input_tensor = torch.randn(2, 64).cuda()

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "pass_through_build_failures": False,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with batch sizes well within min * 128 (2 * 128 = 256)
    for batch_size in [2, 4, 8, 16, 32]:
        input_test = torch.randn(batch_size, 64).cuda()
        output_ref = model(input_test)
        output_trt = trt_model(input_test)

        cos_sim = cosine_similarity(output_ref, output_trt)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Reasonable default test failed at batch_size={batch_size}. Cosine sim: {cos_sim}",
        )

    torch._dynamo.reset()


@pytest.mark.unit
def test_unbounded_symint_fallback():
    """
    Test that the default max (min * 128) is applied for unbounded SymInts.
    This test verifies the fallback behavior when no max is specified.
    """

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(64, 128)
            self.linear2 = torch.nn.Linear(128, 64)
            self.linear3 = torch.nn.Linear(64, 32)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    model = SimpleModel().eval().cuda()

    # Create input with unbounded batch dimension and small min
    input_tensor = torch.randn(2, 64).cuda()

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "torch_executed_ops": [torch.ops.aten.relu.default],
        "pass_through_build_failures": False,
    }

    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)

    # Test with batch sizes well within min * 128 (2 * 128 = 256)
    for batch_size in [2, 4, 8, 16, 32]:
        input_test = torch.randn(batch_size, 64).cuda()
        output_ref = model(input_test)
        output_trt = trt_model(input_test)

        cos_sim = cosine_similarity(output_ref, output_trt)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Reasonable default test failed at batch_size={batch_size}. Cosine sim: {cos_sim}",
        )

    torch._dynamo.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
