import unittest

import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from utils import COSINE_THRESHOLD, cosine_similarity


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    torchtrt.ENABLED_FEATURES.tensorrt_rtx,
    "aten::adaptive_avg_pool2d is implemented via plugins which is not supported for tensorrt_rtx",
)
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

        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(trt_engine)
            with engine.create_execution_context() as ctx:
                out = torch.empty(
                    size=tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
                ).to("cuda:0")
                bindings = [
                    self.input.contiguous().data_ptr(),
                    out.contiguous().data_ptr(),
                ]

                # Assign tensor address appropriately
                for idx in range(engine.num_io_tensors):
                    ctx.set_tensor_address(engine.get_tensor_name(idx), bindings[idx])
                ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

                cos_sim = cosine_similarity(self.model(self.input), out)
                self.assertTrue(
                    cos_sim > COSINE_THRESHOLD,
                    msg=f"TestPyTorchToTRTEngine TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
                )


if __name__ == "__main__":
    unittest.main()
