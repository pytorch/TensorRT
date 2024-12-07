import torch
import torch_tensorrt
import torch_tensorrt as torchtrt
import torchvision

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# load model
model = (
    torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    .eval()
    .to("cuda")
)

# Compile with Torch TensorRT;
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch_tensorrt.dtype.f16},
)

ts_trt_model = torch.jit.trace(trt_model, torch.rand(1, 3, 224, 224).to("cuda"))

# Save the model
torch.jit.save(ts_trt_model, "/triton_example/model_repository/resnet50/1/model.pt")
