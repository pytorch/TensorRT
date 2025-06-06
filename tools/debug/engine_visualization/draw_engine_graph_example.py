import logging
import os

import numpy as np
import torch
import torch_tensorrt as torch_tensorrt
import torchvision.models as models
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR

inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]
model = models.resnet18(pretrained=False).eval().to("cuda")
exp_program = torch.export.export(model, tuple(inputs))

with torch_tensorrt.dynamo.Debugger(
    "graphs",
    logging_dir=DEBUG_LOGGING_DIR,
    capture_fx_graph_after=["constant_fold"],
    save_engine_profile=True,
    profile_format="trex",
    engine_builder_monitor=False,
):
    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=inputs,
        enabled_precisions={torch.float},
        truncate_double=True,
        use_python_runtime=False,
        min_block_size=1,
    )
    trt_output = trt_gm(*inputs)

    from draw_engine_graph import draw_engine

    draw_engine(os.path.join(DEBUG_LOGGING_DIR, "engine_visualization_profile"))
