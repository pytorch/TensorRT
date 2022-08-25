import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import tensorrt as trt


class TestPyTorchToTRTEngine(unittest.TestCase):
    def test_pt_to_trt(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda:0")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda:0")
        self.ts_model = torch.jit.script(self.model)
        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "truncate_long_and_double": True,
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        }

        trt_engine = torchtrt.ts.convert_method_to_trt_engine(
            self.ts_model, "forward", **compile_spec
        )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(trt_engine)
            with engine.create_execution_context() as ctx:
                out = torch.empty(size=tuple(engine.get_binding_shape(1))).to("cuda:0")
                bindings = [
                    self.input.contiguous().data_ptr(),
                    out.contiguous().data_ptr(),
                ]
                ctx.execute_async(
                    batch_size=1,
                    bindings=bindings,
                    stream_handle=torch.cuda.current_stream(
                        device="cuda:0"
                    ).cuda_stream,
                )
                same = (out - self.ts_model(self.input)).abs().max()
                self.assertTrue(same < 2e-3)


if __name__ == "__main__":
    unittest.main()
