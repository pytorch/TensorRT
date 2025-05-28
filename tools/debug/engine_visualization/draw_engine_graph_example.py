import numpy as np
import torch
import torch_tensorrt as torch_tensorrt
import torchvision.models as models

inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]
model = models.resnet18(pretrained=False).eval().to("cuda")
exp_program = torch.export.export(model, tuple(inputs))
enabled_precisions = {torch.float}
debug = False
workspace_size = 20 << 30
min_block_size = 0
use_python_runtime = False
torch_executed_ops = {}
trt_gm = torch_tensorrt.dynamo.compile(
    exp_program,
    inputs=inputs,
    enabled_precisions=enabled_precisions,
    truncate_double=True,
    debug=True,
    use_python_runtime=False,
    engine_vis_dir="/home/profile",
)
trt_output = trt_gm(*inputs)

from draw_engine_graph import draw_engine

draw_engine("/home/profile")
