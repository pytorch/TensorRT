from monai.networks.nets import UNet
import torch
import torch_tensorrt

device = "cuda:0"

# Define the 2D U-Net model
model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    act="relu",
    norm="batch",
    dropout=0.1,
).to(device).half().eval()

# (batch size, channels, height, width)
input_tensor = torch.randn(1, 3, 256, 256, device=device).half()

backend = "torch_tensorrt"

# Compile the model with Torch-TensorRT backend
model = torch.compile(
    model,
    backend=backend,
    options={
        "use_python_runtime": False,
        "enabled_precisions": {torch.float16},
        "truncate_double": True,
        "debug": True,
        "min_block_size": 1,
    },
    dynamic=False,
)

# Perform inference with the compiled model
with torch.no_grad():
    output = model(input_tensor)

print(output)
