import logging
import os

import numpy as np
import torch
import torch_tensorrt as torch_tensorrt
import torchvision.models as models

inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]
model = models.resnet18(pretrained=False).eval().to("cuda")
exp_program = torch.export.export(model, tuple(inputs))
enabled_precisions = {torch.float}
workspace_size = 20 << 30
# min_block_size = 0
use_python_runtime = False
torch_executed_ops = {}
logging_dir = "/home/profile"
with torch_tensorrt.dynamo.Debugger(
    "graphs",
    logging_dir=logging_dir,
    capture_fx_graph_after=["constant_fold"],
    save_engine_profile=True,
):
    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        truncate_double=True,
        use_python_runtime=False,
    )
    trt_output = trt_gm(*inputs)

    from draw_engine_graph import draw_engine

    draw_engine(os.path.join(logging_dir, "engine_visualization"))
print()
