"""
.. _torch_compile_advanced_usage:

Torch Compile Advanced Usage
======================================================

This interactive script is intended as an overview of the process by which `torch_tensorrt.compile(..., ir="torch_compile", ...)` works, and how it integrates with the `torch.compile` API.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt

# %%


# We begin by defining a model
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_out = self.relu(x)
        y_out = self.relu(y)
        x_y_out = x_out + y_out
        return torch.mean(x_y_out)


# %%
# Compilation with `torch.compile` Using Default Settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Define sample float inputs and initialize model
sample_inputs = [torch.rand((5, 7)).cuda(), torch.rand((5, 7)).cuda()]
model = Model().cuda().eval()

# %%

# Next, we compile the model using torch.compile
# For the default settings, we can simply call torch.compile
# with the backend "torch_tensorrt", and run the model on an
# input to cause compilation, as so:
optimized_model = torch.compile(model, backend="torch_tensorrt", dynamic=False)
optimized_model(*sample_inputs)

# %%
# Compilation with `torch.compile` Using Custom Settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# First, we use Torch utilities to clean up the workspace
# after the previous compile invocation
torch._dynamo.reset()

# Define sample half inputs and initialize model
sample_inputs_half = [
    torch.rand((5, 7)).half().cuda(),
    torch.rand((5, 7)).half().cuda(),
]
model_half = Model().cuda().eval()

# %%

# If we want to customize certain options in the backend,
# but still use the torch.compile call directly, we can provide
# custom options to the backend via the "options" keyword
# which takes in a dictionary mapping options to values.
#
# For accepted backend options, see the CompilationSettings dataclass:
# py/torch_tensorrt/dynamo/_settings.py
backend_kwargs = {
    "enabled_precisions": {torch.half},
    "min_block_size": 2,
    "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
    "optimization_level": 4,
    "use_python_runtime": False,
}

# Run the model on an input to cause compilation, as so:
optimized_model_custom = torch.compile(
    model_half,
    backend="torch_tensorrt",
    options=backend_kwargs,
    dynamic=False,
)
optimized_model_custom(*sample_inputs_half)

# %%
# Cleanup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Finally, we use Torch utilities to clean up the workspace
torch._dynamo.reset()

# %%
# Cuda Driver Error Note
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Occasionally, upon exiting the Python runtime after Dynamo compilation with `torch_tensorrt`,
# one may encounter a Cuda Driver Error. This issue is related to https://github.com/NVIDIA/TensorRT/issues/2052
# and can be resolved by wrapping the compilation/inference in a function and using a scoped call, as in::
#
#       if __name__ == '__main__':
#           compile_engine_and_infer()
