# type: ignore
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
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
    "aten::adaptive_avg_pool2d is implemented via plugins which is not supported for tensorrt_rtx",
)
class TestToBackendLowering(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn((1, 3, 300, 300)).to("cuda")
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.spec = {
            "forward": torchtrt.ts.TensorRTCompileSpec(
                **{
                    "inputs": [torchtrt.Input([1, 3, 300, 300])],
                    "enabled_precisions": {torch.float},
                    "refit": False,
                    "device": {
                        "device_type": torchtrt.DeviceType.GPU,
                        "gpu_id": 0,
                        "dla_core": 0,
                        "allow_gpu_fallback": True,
                    },
                    "capability": torchtrt.EngineCapability.STANDARD.to(
                        torchtrt._C.EngineCapability
                    ),
                    "num_avg_timing_iters": 1,
                    "disable_tf32": False,
                }
            )
        }

    def test_to_backend_lowering(self):
        trt_mod = torch._C._jit_to_backend("tensorrt", self.scripted_model, self.spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TestToBackendLowering TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
