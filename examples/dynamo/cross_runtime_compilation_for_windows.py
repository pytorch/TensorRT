"""
.. _resnet_cross_runtime_compilation_for_windows_example:

cross runtime compilation limitations:
The cross compile and saved model can only be loaded in Windows, it can no longer be loaded in Linux
The cross compile and saved model can only be loaded in the same Compute capacity as the Linux which it has been cross compiled

Cross runtime compilation for windows example
======================================================

Compile and save the Resnet Model using Torch-TensorRT in Linux:

python cross_runtime_compilation_for_windows.py --path trt_resnet.ep

Load the Resnet Model saved in Windows:

python cross_runtime_compilation_for_windows.py --path trt_resnet.ep --load True

"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import argparse
import platform

import torch
import torch_tensorrt as torchtrt
import torchvision.models as models

PARSER = argparse.ArgumentParser(
    description="Cross runtime comilation for windows example: Resnet Model"
)
PARSER.add_argument(
    "--load", default=False, type=bool, required=False, help="Load the model in Windows"
)
PARSER.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to the saved model file",
)

args = PARSER.parse_args()
torch.manual_seed(0)
model = models.resnet18().eval().cuda()
input = torch.rand((1, 3, 224, 224)).to("cuda")
inputs = [input]

# %%
# According to the argument, it is either cross compile and save resnet model for windows in Linux
# or load the saved resnet model in Windows
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if args.load:
    # load the saved model in Windows
    if platform.system() != "Windows" or platform.machine() != "AMD64":
        raise ValueError(
            "cross runtime compiled model for windows can only be loaded in Windows system"
        )
    loaded_model = torch.export.load(args.path).module()
    loaded_model(input)
    print(f"model has been successfully loaded from ${args.path}")

else:
    if platform.system() != "Linux" or platform.architecture()[0] != "64bit":
        raise ValueError(
            "cross runtime compiled model for windows can only be compiled in Linux system"
        )
    compile_spec = {
        "min_block_size": 1,
    }
    torchtrt.cross_compile_save_for_windows(
        model, file_path=args.path, inputs=inputs, **compile_spec
    )
    print(
        f"model has been successfully cross compiled and saved in Linux to {args.path}"
    )
