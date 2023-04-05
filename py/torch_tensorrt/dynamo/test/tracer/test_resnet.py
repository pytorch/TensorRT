import unittest

import torch

import torch._dynamo.config
import torchvision
from torch_tensorrt.dynamo.lower import compile
from torch_tensorrt.dynamo.utils import LowerPrecision


class ResnetTest(unittest.TestCase):
    def test_resnet18_aten(self):
        mod = torchvision.models.resnet18()
        mod = mod.cuda().half().eval()

        inputs = [torch.ones(32, 3, 224, 224)]
        inputs = [i.cuda().half() for i in inputs]

        aten_mod = compile(
            mod,
            inputs,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            is_aten=True,
        )
        aten_output = aten_mod(*inputs)
        aten_output = aten_output[0]
        fx_mod = compile(
            mod,
            inputs,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            is_aten=False,
        )
        fx_output = fx_mod(*inputs)
        # Kernel selection is tricky in TRT with big variance as shown below:
        # Mismatched elements: 30816 / 32000 (96.3%)
        # Greatest absolute difference: 0.05859375 at index (0, 499) (up to 1e-05 allowed)
        # Greatest relative difference: 3.293713681986265 at index (0, 142) (up to 0.001 allowed)
        # so we choose to use cosine similarity
        cos_val = torch.nn.functional.cosine_similarity(
            aten_output.flatten(), fx_output.flatten(), dim=0, eps=1e-4
        )
        self.assertTrue(cos_val.detach().cpu().numpy() > 0.999)

    def test_resnet18_aten_dynamic(self):
        mod = torchvision.models.resnet18()
        mod = mod.cuda().half().eval()

        inputs = [torch.ones(32, 3, 224, 224)]
        inputs = [i.cuda().half() for i in inputs]

        aten_mod = compile(
            mod,
            inputs,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            is_aten=True,
        )
        aten_output = aten_mod(*inputs)
        aten_output = aten_output[0]
        fx_mod = compile(
            mod,
            inputs,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            is_aten=False,
        )
        fx_output = fx_mod(*inputs)

        cos_val = torch.nn.functional.cosine_similarity(
            aten_output.flatten(), fx_output.flatten(), dim=0, eps=1e-4
        )
        self.assertTrue(cos_val.detach().cpu().numpy() > 0.999)
