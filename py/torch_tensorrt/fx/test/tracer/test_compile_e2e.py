import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import timm


COS_SIM_THRESHOLD = 0.99


def cosine_similarity_custom(trt_out, torch_out):
    torch.nn.functional.cosine_similarity(
        trt_out.flatten(), torch_out.flatten(), dim=0, eps=1e-6
    )


class TestCompileE2E(unittest.TestCase):
    def test_resnet18_fx(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [self.input],
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.compile(self.model, ir="fx", **compile_spec)
        cos_sim = cosine_similarity_custom(
            self.model(self.input),
            trt_mod(self.input),
            dim=0,
            eps=1e-6,
        )
        self.assertTrue(
            cos_sim > COS_SIM_THRESHOLD,
            msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COS_SIM_THRESHOLD}",
        )

    def test_mobilenet_v2_fx(self):
        self.model = models.mobilenet_v2(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [self.input],
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.compile(self.model, ir="fx", **compile_spec)
        cos_sim = cosine_similarity_custom(
            self.model(self.input),
            trt_mod(self.input),
            dim=0,
            eps=1e-6,
        )
        self.assertTrue(
            cos_sim > COS_SIM_THRESHOLD,
            msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COS_SIM_THRESHOLD}",
        )

    def test_efficientnet_b0_fx(self):
        self.model = (
            timm.create_model("efficientnet_b0", pretrained=True).eval().to("cuda")
        )
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [self.input],
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.compile(self.model, ir="fx", **compile_spec)
        cos_sim = cosine_similarity_custom(
            self.model(self.input),
            trt_mod(self.input),
            dim=0,
            eps=1e-6,
        )
        self.assertTrue(
            cos_sim > COS_SIM_THRESHOLD,
            msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COS_SIM_THRESHOLD}",
        )

    def test_resnet18_half_fx(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda").half()
        self.input = torch.randn((1, 3, 224, 224)).to("cuda").half()

        compile_spec = {
            "inputs": [self.input],
            "enabled_precisions": {torch.half},
        }

        trt_mod = torchtrt.compile(self.model, ir="fx", **compile_spec)
        cos_sim = cosine_similarity_custom(
            self.model(self.input),
            trt_mod(self.input),
            dim=0,
            eps=1e-6,
        )
        self.assertTrue(
            cos_sim > COS_SIM_THRESHOLD,
            msg=f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COS_SIM_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
