from copy import deepcopy

import numpy as np
import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning

from ..testing_utilities import lower_graph_testing

# This testcase assumes that torch.ops.aten.clamp.default converter doesn't support
# dynamic shapes. One should remove this testcase when the support is added.
# This testcase tests if the graph is partitioned correctly into a TRT segment
# and a Pytorch segment when the torch.ops.aten.clamp.default converter gets disabled
# due to lack of dynamic shape support.


class TestDynamicPartitioning(TestCase):
    def test_partition_dynamic_clamp(self):
        class Clamp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                return torch.ops.aten.clamp.default(x, min=2.5, max=6.5)

        model = Clamp().eval().cuda()
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 8, 8),
                    opt_shape=(4, 3, 8, 8),
                    max_shape=(8, 3, 8, 8),
                    dtype=torch.float32,
                    name="x",
                )
            ],
            dryrun=True,
            min_block_size=1,
        )
        trt_segments, pyt_segments = 0, 0
        for submod in list(trt_model.named_children()):
            if "_run_on_acc" in submod[0]:
                trt_segments += 1
            elif "_run_on_gpu" in submod[0]:
                pyt_segments += 1

        self.assertEquals(
            trt_segments,
            1,
            f"Number of TRT segments should be 1 but got {trt_segments}",
        )
        self.assertEquals(
            pyt_segments,
            1,
            f"Number of PyTorch segments should be 1 but got {pyt_segments}",
        )

    def test_assume_dynamic_shape_support_converters(self):
        class Clamp(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                return torch.ops.aten.clamp.default(x, min=2.5, max=6.5)

        model = Clamp().eval().cuda()
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 3, 8, 8),
                    opt_shape=(4, 3, 8, 8),
                    max_shape=(8, 3, 8, 8),
                    dtype=torch.float32,
                    name="x",
                )
            ],
            dryrun=True,
            assume_dynamic_shape_support=True,
            min_block_size=1,
        )

        trt_segments, pyt_segments = 0, 0
        for submod in list(trt_model.named_children()):
            if "_run_on_acc" in submod[0]:
                trt_segments += 1
            elif "_run_on_gpu" in submod[0]:
                pyt_segments += 1

        self.assertEquals(
            trt_segments,
            1,
            f"Number of TRT segments should be 2 but got {trt_segments}",
        )
        self.assertEquals(
            pyt_segments,
            0,
            f"Number of PyTorch segments should be 0 but got {pyt_segments}",
        )


if __name__ == "__main__":
    run_tests()
