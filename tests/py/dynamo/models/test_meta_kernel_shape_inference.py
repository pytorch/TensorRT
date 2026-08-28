"""
Test meta kernel shape inference by running TRT modules in fake mode.

Each test exports a model, compiles with TRT, then runs the TRT module in fake
mode to verify the meta kernel correctly infers symbolic output shapes.

The test approach:
1. Export a model with dynamic shapes to get an exported program with symbolic SymInts
2. Compile the exported program with Torch-TensorRT
3. Extract the symbolic fake input from the exported program
4. Run both the exported program and TRT module with the same symbolic fake input
5. Verify that both produce the same symbolic output shapes

Currently, tests with dynamic shapes are marked as xfail because the meta kernel
does not preserve symbolic SymInt dimensions - it creates new unbacked symints instead
of reusing the input SymInts. This is a known limitation.
"""

import pytest
import sympy
import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.export import Dim
from torch_tensorrt.dynamo.runtime.meta_ops.register_meta_ops import (
    _apply_symbolic_shape_expressions,
)


class TestMetaKernelShapeInference:
    """Test meta kernel by running TRT modules in fake mode"""

    def _test_in_fake_mode(self, model, test_input, dynamic_shapes=None):
        """
        Helper that exports model, compiles with TRT, runs in fake mode.
        Returns (exported_output, trt_output, fake_input) for shape comparison.
        """
        # Export with dynamic shapes
        if dynamic_shapes:
            exported = torch.export.export(
                model, args=(test_input,), dynamic_shapes=dynamic_shapes
            )
        else:
            exported = torch.export.export(model, args=(test_input,))

        # Compile with TRT
        compiled = torch_tensorrt.compile(
            exported, inputs=[test_input], min_block_size=1
        )

        # Get the fake input from the exported program - it has symbolic shapes
        from torch._guards import detect_fake_mode

        fake_input = None
        for node in exported.graph.nodes:
            if node.op == "placeholder" and node.name == "x" and "val" in node.meta:
                fake_input = node.meta["val"]
                break

        assert (
            fake_input is not None
        ), "Could not find input placeholder 'x' in exported program"

        # Get the fake mode
        fake_mode = detect_fake_mode((fake_input,))
        assert fake_mode is not None, "Could not detect fake mode from exported program"

        # Run both exported and compiled in the same fake mode
        with fake_mode:
            exported_output = exported.module()(fake_input)
            trt_output = compiled(fake_input)

            return exported_output, trt_output, fake_input

    def test_identity_static(self):
        """Test meta kernel with static shapes (identity operation)"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 64, 64).cuda()

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input
        )

        print(f"Input shape: {fake_input.shape}")
        print(f"Exported output shape: {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Both should produce same shape
        assert exported_output.shape == trt_output.shape
        assert trt_output.shape == (4, 64, 64, 64)

    def test_downsample_static(self):
        """Test meta kernel with static shapes (stride=2 downsampling)"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 64, 64).cuda()

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input
        )

        print(f"Input shape: {fake_input.shape}")
        print(f"Exported output shape: {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Both should produce same downsampled shape
        assert exported_output.shape == trt_output.shape
        assert trt_output.shape == (4, 64, 32, 32)

    def test_dynamic_batch(self):
        """Test meta kernel preserves symbolic batch dimension"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 64, 64).cuda()

        batch = Dim("batch", min=1, max=8)
        dynamic_shapes = {"x": {0: batch}}

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input, dynamic_shapes
        )

        print(f"Input shape: {fake_input.shape}")
        print(f"Exported output shape: {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Both should have symbolic batch
        assert isinstance(
            fake_input.shape[0], torch.SymInt
        ), "Input batch should be symbolic"
        assert isinstance(
            exported_output.shape[0], torch.SymInt
        ), "Exported output batch should be symbolic"
        assert isinstance(
            trt_output.shape[0], torch.SymInt
        ), "TRT output batch should be symbolic"

        # Shapes should match
        assert exported_output.shape == trt_output.shape

    def test_arithmetic_h_div_2(self):
        """Test meta kernel infers h//2 symbolic relationship"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv(x))
                h = x.shape[2]
                return x[:, :, : h // 2, :]

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 64, 64).cuda()

        batch = Dim("batch", min=1, max=8)
        h_base = Dim("h_base", min=16, max=64)
        w_base = Dim("w_base", min=16, max=64)
        dynamic_shapes = {"x": {0: batch, 2: 2 * h_base, 3: 2 * w_base}}

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input, dynamic_shapes
        )

        print(f"Input shape (height=2*h_base): {fake_input.shape}")
        print(f"Exported output shape (height=h_base): {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Height should be symbolic and correctly inferred
        assert isinstance(
            fake_input.shape[2], torch.SymInt
        ), "Input height should be symbolic"
        assert isinstance(
            exported_output.shape[2], torch.SymInt
        ), "Exported output height should be symbolic"
        assert isinstance(
            trt_output.shape[2], torch.SymInt
        ), "TRT output height should be symbolic"

        # Shapes should match
        assert exported_output.shape == trt_output.shape

    def test_stride_2_dynamic(self):
        """Test meta kernel infers h//2 from stride=2 convolution"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 64, 64).cuda()

        batch = Dim("batch", min=1, max=8)
        h_base = Dim("h_base", min=16, max=64)
        w_base = Dim("w_base", min=16, max=64)
        # Input must be even for stride=2
        dynamic_shapes = {"x": {0: batch, 2: 2 * h_base, 3: 2 * w_base}}

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input, dynamic_shapes
        )

        print(f"Input shape (2*h_base): {fake_input.shape}")
        print(f"Exported output shape (h_base): {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Height should be symbolic
        assert isinstance(
            fake_input.shape[2], torch.SymInt
        ), "Input height should be symbolic"
        assert isinstance(
            exported_output.shape[2], torch.SymInt
        ), "Exported output height should be symbolic"
        assert isinstance(
            trt_output.shape[2], torch.SymInt
        ), "TRT output height should be symbolic"

        # Shapes should match
        assert exported_output.shape == trt_output.shape

    def test_concat(self):
        """Test meta kernel with concat operation (concatenates on height dimension)"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv(x))
                # Concatenate x with itself on height dimension to double the height
                return torch.cat([x, x], dim=2)

        model = Model().eval().cuda()
        test_input = torch.randn(4, 3, 32, 32).cuda()

        batch = Dim("batch", min=1, max=8)
        h = Dim("h", min=16, max=64)
        w = Dim("w", min=16, max=64)
        dynamic_shapes = {"x": {0: batch, 2: h, 3: w}}

        exported_output, trt_output, fake_input = self._test_in_fake_mode(
            model, test_input, dynamic_shapes
        )

        print(f"Input shape: {fake_input.shape}")
        print(f"Exported output shape (2*h): {exported_output.shape}")
        print(f"TRT output shape: {trt_output.shape}")

        # Height should be symbolic (2x input from concat)
        assert isinstance(
            fake_input.shape[2], torch.SymInt
        ), "Input height should be symbolic"
        assert isinstance(
            exported_output.shape[2], torch.SymInt
        ), "Exported output height should be symbolic"
        assert isinstance(
            trt_output.shape[2], torch.SymInt
        ), "TRT output height should be symbolic"

        # Shapes should match
        assert exported_output.shape == trt_output.shape


class TestApplySymbolicShapeExpressions:
    """CPU-only tests for remapping serialized expressions into a fresh ShapeEnv."""

    @staticmethod
    def _shape_info(input_expr, output_expr):
        return {
            "inputs": [
                {
                    "shape_exprs": [input_expr],
                    "dtype": torch.float32,
                    "name": "x",
                }
            ],
            "outputs": [
                {
                    "shape_exprs": [output_expr],
                    "dtype": torch.float32,
                }
            ],
        }

    def test_output_symbol_does_not_alias_same_named_runtime_symbol(self):
        compile_input = sympy.Symbol("s0", integer=True)
        compile_output = sympy.Symbol("u0", integer=True)
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env):
            runtime_input = shape_env.create_unbacked_symint()
            shape_env._constrain_range_for_size(runtime_input.node.expr)
            fake_input = torch.empty(runtime_input)
            output = _apply_symbolic_shape_expressions(
                [fake_input], self._shape_info(compile_input, compile_output)
            )[0]

        assert output.shape[0].node.expr != runtime_input.node.expr

    def test_output_only_symbol_uses_fake_mode_shape_env_and_is_size_like(self):
        compile_output = sympy.Symbol("u0", integer=True)
        shape_env = ShapeEnv()
        shape_info = {
            "inputs": [
                {
                    "shape_exprs": [16],
                    "dtype": torch.float32,
                    "name": "x",
                }
            ],
            "outputs": [
                {
                    "shape_exprs": [compile_output],
                    "dtype": torch.float32,
                }
            ],
        }

        with FakeTensorMode(shape_env=shape_env):
            output = _apply_symbolic_shape_expressions([torch.empty(16)], shape_info)[0]
            output_dim = output.shape[0]
            query = 1 > output_dim + torch.sym_max(0, 1 - output_dim)

        assert output_dim.node.shape_env is shape_env
        assert output_dim.node.expr in shape_env.size_like
        assert not bool(shape_env.evaluate_expr(query.node.expr, size_oblivious=True))

    def test_derived_expression_uses_runtime_input_symbol(self):
        compile_input = sympy.Symbol("s0", integer=True)
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env):
            runtime_input = shape_env.create_unbacked_symint()
            shape_env._constrain_range_for_size(runtime_input.node.expr)
            fake_input = torch.empty(runtime_input)
            output = _apply_symbolic_shape_expressions(
                [fake_input],
                self._shape_info(compile_input, 2 * compile_input + 1),
            )[0]

        assert (
            sympy.simplify(
                output.shape[0].node.expr - (2 * runtime_input.node.expr + 1)
            )
            == 0
        )

    def test_composite_input_expression_is_solved_in_runtime_namespace(self):
        compile_base = sympy.Symbol("s0", integer=True)
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env):
            runtime_base = shape_env.create_unbacked_symint()
            shape_env._constrain_range_for_size(runtime_base.node.expr)
            fake_input = torch.empty(2 * runtime_base)
            output = _apply_symbolic_shape_expressions(
                [fake_input],
                self._shape_info(2 * compile_base, compile_base),
            )[0]

        assert output.shape[0].node.expr == runtime_base.node.expr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
