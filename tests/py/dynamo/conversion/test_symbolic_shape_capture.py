import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion._symbolic_shape_capture import (
    extract_symbolic_shape_expressions,
)


class TestSymbolicShapeCaptureScalarDtype(TestCase):
    """A SymFloat scalar input has no dtype of its own in FX metadata; the
    recorded dtype must match the engine's actual binding (float32)."""

    def _make_symfloat_input_module(self) -> torch.fx.GraphModule:
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            scale = shape_env.create_unbacked_symfloat()
            x = torch.empty((4,), dtype=torch.float32)

        graph = torch.fx.Graph()
        scale_input = graph.placeholder("scale")
        scale_input.meta["val"] = scale
        x_input = graph.placeholder("x")
        x_input.meta["val"] = x
        output = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(x_input, scale_input)
        )
        output.meta["val"] = x
        graph.output(output)
        return torch.fx.GraphModule({}, graph)

    def test_symfloat_input_uses_engine_binding_dtype(self):
        module = self._make_symfloat_input_module()
        engine_inputs = [
            Input([1], dtype=torch.float32, name="scale"),
            Input((4,), dtype=torch.float32, name="x"),
        ]

        metadata = extract_symbolic_shape_expressions(module, inputs=engine_inputs)

        scale_info = next(
            info for info in metadata["inputs"] if info["name"] == "scale"
        )
        self.assertTrue(scale_info["is_scalar"])
        self.assertEqual(scale_info["dtype"], torch.float32)

    def test_symfloat_input_defaults_to_float32_without_engine_inputs(self):
        # Used to hardcode float64 -- no real engine binding is ever float64.
        module = self._make_symfloat_input_module()

        metadata = extract_symbolic_shape_expressions(module)

        scale_info = next(
            info for info in metadata["inputs"] if info["name"] == "scale"
        )
        self.assertEqual(scale_info["dtype"], torch.float32)

    def test_ordinary_input_without_explicit_dtype_does_not_raise(self):
        # An Input with dtype omitted (dtype.unknown) used to raise TypeError.
        module = self._make_symfloat_input_module()
        engine_inputs = [
            Input([1], dtype=torch.float32, name="scale"),
            Input((4,)),  # dtype intentionally omitted
        ]

        metadata = extract_symbolic_shape_expressions(module, inputs=engine_inputs)

        scale_info = next(
            info for info in metadata["inputs"] if info["name"] == "scale"
        )
        self.assertEqual(scale_info["dtype"], torch.float32)


if __name__ == "__main__":
    run_tests()
