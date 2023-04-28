# Owner(s): ["oncall: gpu_enablement"]

import io
import os

import torch
import torch.fx

import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule

from torch_tensorrt import TRTModuleNext
from torch_tensorrt import Device
from torch_tensorrt.fx.utils import LowerPrecision


class TestTRTModule(TestCase):
    def test_save_and_load_trt_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(mod, input_specs=InputTensorSpec.from_tensors(inputs))
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        torch.save(trt_mod, "trt.pt")
        reload_trt_mod = torch.load("trt.pt")

        torch.testing.assert_close(
            reload_trt_mod(inputs[0].cuda()).cpu(), ref_output, rtol=1e-04, atol=1e-04
        )
        os.remove(f"{os.getcwd()}/trt.pt")

    def test_save_and_load_state_dict(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(mod, input_specs=InputTensorSpec.from_tensors(inputs))
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        st = trt_mod.state_dict()

        new_trt_mod = TRTModule()
        new_trt_mod.load_state_dict(st)

        torch.testing.assert_close(
            new_trt_mod(inputs[0].cuda()).cpu(), ref_output, rtol=1e-04, atol=1e-04
        )


class TestTRTModuleInt64Input(TestCase):
    def test_save_and_load_trt_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(5, 5).long()]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            truncate_long_and_double=True,
        )
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        torch.save(trt_mod, "trt.pt")
        reload_trt_mod = torch.load("trt.pt")

        torch.testing.assert_close(
            reload_trt_mod(inputs[0].cuda()).cpu(),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
            check_dtype=False,
        )
        os.remove(f"{os.getcwd()}/trt.pt")

    def test_save_and_load_state_dict(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(5, 5).long()]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            truncate_long_and_double=True,
        )
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        st = trt_mod.state_dict()

        new_trt_mod = TRTModule()
        new_trt_mod.load_state_dict(st)

        torch.testing.assert_close(
            new_trt_mod(inputs[0].cuda()).cpu(),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
            check_dtype=False,
        )


class TestTRTModuleFloat64Input(TestCase):
    def test_save_and_load_trt_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(5, 5).double()]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            truncate_long_and_double=True,
        )
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        torch.save(trt_mod, "trt.pt")
        reload_trt_mod = torch.load("trt.pt")

        torch.testing.assert_close(
            reload_trt_mod(inputs[0].cuda()).cpu(),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
            check_dtype=False,
        )
        os.remove(f"{os.getcwd()}/trt.pt")

    def test_save_and_load_state_dict(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(5, 5).double()]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            truncate_long_and_double=True,
        )
        trt_mod = TRTModule(*interp.run(lower_precision=LowerPrecision.FP32))
        st = trt_mod.state_dict()

        new_trt_mod = TRTModule()
        new_trt_mod.load_state_dict(st)

        torch.testing.assert_close(
            new_trt_mod(inputs[0].cuda()).cpu(),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
            check_dtype=False,
        )


class TestTRTModuleNext(TestCase):
    def test_save_and_load_trt_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)

        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            explicit_batch_dimension=True,
        )
        interp_res = interp.run(lower_precision=LowerPrecision.FP32)

        with io.BytesIO() as engine_bytes:
            engine_bytes.write(interp_res.engine.serialize())
            engine_str = engine_bytes.getvalue()

        trt_mod = TRTModuleNext(
            name="TestModule",
            serialized_engine=engine_str,
            input_binding_names=interp_res.input_names,
            output_binding_names=interp_res.output_names,
            target_device=Device(f"cuda:{torch.cuda.current_device()}"),
        )

        torch.save(trt_mod, "trt.pt")
        reload_trt_mod = torch.load("trt.pt")

        torch.testing.assert_allclose(
            reload_trt_mod(inputs[0].cuda()).cpu().reshape_as(ref_output),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
        )
        os.remove(f"{os.getcwd()}/trt.pt")

    def test_save_and_load_state_dict(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch.randn(1, 1)]
        mod = TestModule().eval()
        ref_output = mod(*inputs)

        mod = acc_tracer.trace(mod, inputs)
        interp = TRTInterpreter(
            mod,
            input_specs=InputTensorSpec.from_tensors(inputs),
            explicit_batch_dimension=True,
        )
        interp_res = interp.run(lower_precision=LowerPrecision.FP32)

        with io.BytesIO() as engine_bytes:
            engine_bytes.write(interp_res.engine.serialize())
            engine_str = engine_bytes.getvalue()

        trt_mod = TRTModuleNext(
            name="TestModule",
            serialized_engine=engine_str,
            input_binding_names=interp_res.input_names,
            output_binding_names=interp_res.output_names,
            target_device=Device(f"cuda:{torch.cuda.current_device()}"),
        )

        st = trt_mod.state_dict()

        new_trt_mod = TRTModuleNext()
        new_trt_mod.load_state_dict(st)

        torch.testing.assert_allclose(
            new_trt_mod(inputs[0].cuda()).cpu().reshape_as(ref_output),
            ref_output,
            rtol=1e-04,
            atol=1e-04,
        )


if __name__ == "__main__":
    run_tests()
