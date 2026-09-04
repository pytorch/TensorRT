import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

INPUT_SHAPE = (8, 16)


class Scale(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2.0)


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestEngineFailureDiagnostics(TestCase):
    def setUp(self):
        self.addCleanup(torch._dynamo.reset)
        compiled = torch_tensorrt.compile(
            Scale().eval().cuda(),
            ir="dynamo",
            inputs=(torch.randn(INPUT_SHAPE, device="cuda"),),
            min_block_size=1,
            pass_through_build_failures=True,
            use_python_runtime=False,
        )
        engines = [
            module
            for module in compiled.modules()
            if isinstance(module, TorchTensorRTModule)
        ]
        self.assertEqual(len(engines), 1)
        self.engine_module = engines[0]
        self.binding = self.engine_module.input_binding_names[0]
        # The engine's own name is this module's name plus an "_engine" suffix, so
        # the module name is what the message has to carry to identify the engine.
        self.assertNotEqual(self.engine_module.name, "")

    def test_input_dtype_mismatch_names_the_engine_and_the_binding(self):
        wrong_dtype = torch.randint(0, 4, INPUT_SHAPE, dtype=torch.int32, device="cuda")

        with self.assertRaises(Exception) as caught:
            self.engine_module(wrong_dtype)

        message = str(caught.exception)
        self.assertIn(self.engine_module.name, message)
        self.assertIn(self.binding, message)

    def test_input_shape_mismatch_reports_the_rejected_and_declared_shapes(self):
        wrong_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1] + 1)

        with self.assertRaises(Exception) as caught:
            self.engine_module(torch.randn(wrong_shape, device="cuda"))

        message = str(caught.exception)
        self.assertIn(self.engine_module.name, message)
        self.assertIn(self.binding, message)
        self.assertIn(str(list(wrong_shape)), message)
        self.assertIn(str(list(INPUT_SHAPE)), message)


if __name__ == "__main__":
    run_tests()
