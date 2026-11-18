import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

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

    def test_infer_outputs_refuses_a_rejected_shape(self):
        wrong_shape = [INPUT_SHAPE[0], INPUT_SHAPE[1] + 1]

        with self.assertRaises(Exception) as caught:
            self.engine_module.engine.infer_outputs([wrong_shape])

        message = str(caught.exception)
        self.assertIn(self.engine_module.name, message)
        self.assertIn(str(wrong_shape), message)
        self.assertIn(str(list(INPUT_SHAPE)), message)

    def test_failed_engine_assertion_names_the_engine(self):
        import tensorrt as trt

        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        flags = 0
        if not torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        a = network.add_input("a", trt.float32, (-1,))
        b = network.add_input("b", trt.float32, (-1,))
        same = network.add_elementwise(
            network.add_shape(a).get_output(0),
            network.add_shape(b).get_output(0),
            trt.ElementWiseOperation.EQUAL,
        )
        network.add_assertion(same.get_output(0), "a and b differ in length")
        out = network.add_elementwise(a, a, trt.ElementWiseOperation.SUM).get_output(0)
        out.name = "out"
        network.mark_output(out)
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        for name in ("a", "b"):
            profile.set_shape(name, (1,), (4,), (8,))
        config.add_optimization_profile(profile)
        plan = bytes(builder.build_serialized_network(network, config))

        # An engine that needs the output allocator refuses to run under CUDA graphs,
        # and an earlier test in the same process can leave that mode on.
        self.addCleanup(torch_tensorrt.runtime.set_cudagraphs_mode, False)
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

        # The assertion fails while TensorRT infers the shapes, before the engine runs.
        for requires_output_allocator in (False, True):
            with self.subTest(requires_output_allocator=requires_output_allocator):
                engine = TorchTensorRTModule(
                    plan,
                    input_binding_names=["a", "b"],
                    output_binding_names=["out"],
                    name="assertion_check",
                    requires_output_allocator=requires_output_allocator,
                )
                with self.assertRaises(Exception) as caught:
                    engine(torch.ones(3, device="cuda"), torch.ones(2, device="cuda"))
                self.assertIn("assertion_check", str(caught.exception))


class TestPythonEngineRejectedShape(TestCase):
    def setUp(self):
        self.x = torch.randn(INPUT_SHAPE, device="cuda")
        rows = torch.export.Dim("rows", max=16)
        program = torch.export.export(
            Scale().eval().cuda(), (self.x,), dynamic_shapes=({0: rows},)
        )
        compiled = torch_tensorrt.dynamo.compile(
            program,
            inputs=[self.x],
            min_block_size=1,
            pass_through_build_failures=True,
        )
        (module,) = [
            m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)
        ]
        # A build with the C++ runtime backs the module with the C++ engine, so
        # drive the Python engine directly from the same serialized info.
        self.engine = TRTEngine(module._pack_engine_info())
        # The engine's shape range starts at 1 row, so 0 rows is refused.
        self.empty = torch.randn(0, INPUT_SHAPE[1], device="cuda")

    def test_rejected_input_shape_raises(self):
        # TensorRT keeps the last shape it accepted, so a valid call comes first.
        self.engine.execute([self.x])

        with self.assertRaises(RuntimeError) as caught:
            self.engine.execute([self.empty])
        message = str(caught.exception)
        self.assertIn(str(list(self.empty.shape)), message)
        self.assertIn(str([-1, INPUT_SHAPE[1]]), message)
        with self.assertRaisesRegex(RuntimeError, "refused"):
            self.engine.infer_outputs([list(self.empty.shape)])

        self.assertEqual(self.engine.execute([self.x]).shape, self.x.shape)

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
        "The Python fake kernel is registered only without the C++ runtime",
    )
    def test_fake_kernel_reports_a_rejected_shape(self):
        from torch_tensorrt.dynamo.runtime._TRTEngine import execute_engine_fake

        with self.assertRaisesRegex(RuntimeError, "refused"):
            execute_engine_fake([self.empty], self.engine)


if __name__ == "__main__":
    run_tests()
