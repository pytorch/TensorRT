import unittest

import torch
import torch_tensorrt
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT

INPUT_SIZE = (10, 10, 10)
TRIALS = 10


class TestStreams(TestCase):

    def test_non_default_stream_exec(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x + 2) * 7, dim=0)

        with torch.inference_mode():
            dtype = torch.half
            device = torch.device("cuda", 0)
            model = SampleModel().eval().to(device)
            inputs = [torch_tensorrt.Input(shape=(1, 3, 5), dtype=dtype)]

            optimized_model = torch_tensorrt.compile(
                model,
                ir="dynamo",
                inputs=inputs,
                enabled_precisions={dtype},
                min_block_size=1,
                device=device,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )

            for i in range(100):
                new_input = torch.randn((1, 3, 5), dtype=dtype, device=device)

                eager_output = model(new_input)

                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device=device))
                with torch.cuda.stream(stream):
                    trt_output_with_stream = optimized_model(new_input)
                torch.cuda.current_stream(device=device).wait_stream(stream)

                trt_output_without_stream = optimized_model(new_input)

                max_diff_w_stream = float(
                    torch.max(torch.abs(eager_output - trt_output_with_stream))
                )
                max_diff_wo_stream = float(
                    torch.max(torch.abs(eager_output - trt_output_without_stream))
                )
                self.assertAlmostEqual(
                    max_diff_w_stream,
                    0,
                    DECIMALS_OF_AGREEMENT,
                    msg=f"Output using a non default calling stream does not match original model (trial: {i})",
                )
                self.assertAlmostEqual(
                    max_diff_wo_stream,
                    0,
                    DECIMALS_OF_AGREEMENT,
                    msg=f"Output using default stream as calling stream does not match original model (trial: {i})",
                )
