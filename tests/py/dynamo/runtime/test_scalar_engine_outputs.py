import unittest

import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias

ROWS = 16


class ScalarSecondOutput(torch.nn.Module):
    """Returns a tensor and a symbolic int, so the engine has a scalar output."""

    def forward(self, x):
        y = x * 2.0
        return y.sum(), y.shape[0]


class UnitTensorSecondOutput(torch.nn.Module):
    """Returns a tensor and a genuine shape-[1] tensor, whose rank must be left alone."""

    def forward(self, x):
        y = x * 2.0
        return y.sum(), y[:1]


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestScalarEngineOutputs(TestCase):
    def _compile_to_single_engine(self, model):
        self.addCleanup(torch._dynamo.reset)
        inputs = (torch.randn((ROWS,), device="cuda"),)
        rows = torch.export.Dim("rows", min=2, max=4 * ROWS)
        exported = torch.export.export(
            model.eval().cuda(), inputs, dynamic_shapes={"x": {0: rows}}
        )
        gm = torch_tensorrt.dynamo.compile(
            exported,
            inputs=inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=False,
        )
        engines = [
            module for module in gm.modules() if isinstance(module, TorchTensorRTModule)
        ]
        self.assertEqual(len(engines), 1)
        return engines[0]

    def _meta_and_real_output_shapes(self, engine_module):
        real = [
            tuple(output.shape)
            for output in engine_module(torch.randn((ROWS,), device="cuda"))
        ]
        # The meta kernel answers from the engine's recorded shape metadata, which
        # is the only account of the engine any later trace sees.
        with FakeTensorMode(shape_env=ShapeEnv(), allow_non_fake_inputs=True):
            meta = [
                tuple(output.shape)
                for output in engine_module(
                    torch.empty((ROWS,), dtype=torch.float32, device="cuda")
                )
            ]
        return meta, real

    def test_meta_kernel_and_engine_agree_on_scalar_output(self):
        engine_module = self._compile_to_single_engine(ScalarSecondOutput())

        meta, real = self._meta_and_real_output_shapes(engine_module)

        self.assertEqual(meta, real)

    def test_scalar_output_is_bound_at_rank_zero(self):
        engine_module = self._compile_to_single_engine(ScalarSecondOutput())
        recorded = engine_module.symbolic_shape_expressions["outputs"]
        scalar_position = 1
        self.assertTrue(recorded[scalar_position].get("is_scalar", False))
        self.assertEqual(recorded[scalar_position]["shape_exprs"], [])

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_module.serialized_engine)

        self.assertIsNotNone(engine)
        binding = engine_module.output_binding_names[scalar_position]
        self.assertEqual(len(engine.get_tensor_shape(binding)), 0)

    def test_unit_length_tensor_output_keeps_its_rank(self):
        engine_module = self._compile_to_single_engine(UnitTensorSecondOutput())

        meta, real = self._meta_and_real_output_shapes(engine_module)

        self.assertEqual(meta, real)
        self.assertEqual(real[1], (1,))


if __name__ == "__main__":
    run_tests()
