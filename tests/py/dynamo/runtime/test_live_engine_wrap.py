import unittest

import torch
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


def _first_trt_module(mod: torch.nn.Module) -> TorchTensorRTModule:
    for child in mod.modules():
        if isinstance(child, TorchTensorRTModule):
            return child
    raise AssertionError("Compiled module has no TorchTensorRTModule")


class TestLiveEngineWrap(TestCase):
    def test_compile_wraps_live_engine_without_serialized_bytes(self):
        class Add(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.add(a, b)

        model = Add().eval().cuda()
        a = torch.randn((2, 4), device="cuda")
        b = torch.randn((2, 4), device="cuda")

        compiled = torchtrt.compile(
            model,
            inputs=[a, b],
            ir="dynamo",
            min_block_size=1,
            pass_through_build_failures=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

        trt_mod = _first_trt_module(compiled)
        self.assertIsNone(trt_mod.serialized_engine)
        self.assertIsNotNone(trt_mod._live_cuda_engine)
        self.assertIsInstance(trt_mod.engine, TRTEngine)
        self.assertIs(trt_mod.engine.cuda_engine, trt_mod._live_cuda_engine)

        cos_sim = cosine_similarity(model(a, b), compiled(a, b))
        self.assertGreater(cos_sim, COSINE_THRESHOLD)

        packed = trt_mod.get_extra_state()
        self.assertIsNotNone(packed[1])
        self.assertTrue(trt_mod.serialized_engine)

        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
