"""

.. _low_cpu_memory_compilation:

Low CPU Memory Compilation Example
==================================

This example demonstrates compiling a model with a bounded CPU (host) memory
budget using Torch-TensorRT Dynamo. Limiting host RAM use is helpful on
memory-constrained machines or when compiling very large models.

Key notes:
- The toy model below has roughly 430 MB of parameters. We set the CPU
  memory budget to 2 GiB. At compile time, only about 900 MB of host RAM
  may remain available. We expect at most 403 * 4 = 1612 MB of memory to be used by the model.
  So the model is partitioned into two subgraphs to fit the memory budget.

- Performance impact varies by model. When the number of TensorRT engines
  created is small, the impact is typically minimal.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.conversion import CompilationSettings


class net(nn.Module):
    def __init__(self):
        super().__init__()
        # Intentionally large layers to stress host memory during compilation.
        self.conv1 = nn.Conv2d(1024, 4096, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4096)
        self.conv2 = nn.Conv2d(4096, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024 * 56 * 56, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = torch.flatten(x, 1)
        return self.fc1(x)


model = net().eval()
model.to("cuda")
inputs = [torch.randn((1, 1024, 224, 224)).to("cuda")]

enabled_precisions = {torch.float}
use_python_runtime = False

compilation_options = {
    "use_python_runtime": use_python_runtime,
    "enabled_precisions": enabled_precisions,
    "min_block_size": 1,
    "immutable_weights": True,
    "reuse_cached_engines": False,
    "cpu_memory_budget": 2 * 1024 * 1024 * 1024,  # 2 GiB in bytes
}

settings = CompilationSettings(**compilation_options)
with torchtrt.dynamo.Debugger(
    log_level="debug",
    logging_dir="/home/profile/logging/moe",
    engine_builder_monitor=False,
):

    exp_program = torch.export.export(model, tuple(inputs))
    trt_gm = torchtrt.dynamo.compile(
        exp_program,
        inputs=inputs,
        **compilation_options,
    )

    # Expect two back-to-back TensorRT engines due to partitioning under the memory budget.
    print(trt_gm)


"""
You should be able to see two back-to-back TensorRT engines in the graph
Graph Structure:

   Inputs: List[Tensor: (1, 1024, 224, 224)@float32]
    ...
    TRT Engine #1 - Submodule name: _run_on_acc_0
     Engine Inputs: List[Tensor: (1, 1024, 224, 224)@float32]
     Number of Operators in Engine: 9
     Engine Outputs: List[Tensor: (1, 1024, 112, 112)@float32]
    ...
    TRT Engine #2 - Submodule name: _run_on_acc_1
     Engine Inputs: List[Tensor: (1, 1024, 112, 112)@float32]
     Number of Operators in Engine: 3
     Engine Outputs: List[Tensor: (1, 10)@float32]
    ...
   Outputs: List[Tensor: (1, 10)@float32]


GraphModule(
  (_run_on_acc_0): TorchTensorRTModule()
  (_run_on_acc_1): TorchTensorRTModule()
)
"""
