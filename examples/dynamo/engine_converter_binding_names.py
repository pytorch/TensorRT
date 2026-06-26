"""
.. _engine_converter_binding_names:

Naming Engine Bindings with ``convert_exported_program_to_serialized_trt_engine``
=================================================================================

When you use ``torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine``
to produce a raw serialized TensorRT engine, the engine's binding names are
determined by Torch-TensorRT's default policy:

* **Inputs** get the FX placeholder names from the exported program (typically
  the names of your ``forward()`` arguments).
* **Outputs** get auto-generated names ``output0``, ``output1``, etc.

Many production runtimes (Triton Inference Server, custom C++ harnesses,
ONNX-style integrations) bind tensors by name rather than position, and the
auto-generated names often don't line up with what the rest of the serving
stack expects.  The engine converter exposes three keyword arguments that
let you supply binding names shaped like your model's inputs and return
value:

* ``arg_input_binding_names`` — pytree of strings matching ``arg_inputs``
* ``kwarg_input_binding_names`` — pytree of strings matching ``kwarg_inputs``
* ``output_binding_names`` — pytree of strings matching the model's return

The shape of each kwarg directly mirrors how you already pass the values
themselves: ``arg_input_binding_names`` lines up with ``arg_inputs``,
``kwarg_input_binding_names`` lines up with ``kwarg_inputs``.

A note on "the return shape"
----------------------------

A Python function always returns exactly one value.  ``return a, b`` is a
single tuple-shaped return value; ``return {"x": a, "y": b}`` is a single
dict-shaped return value.  Whatever that value is, the exported program
captures it as a pytree.  Its *leaves* — the individual tensors at the
bottom of the structure — become engine bindings, and you supply names in
the same pytree shape.  Inputs work the same way: ``arg_inputs`` is itself
a pytree (a tuple of positional values, each of which can be a tensor or
a nested collection of tensors); ``kwarg_inputs`` is a dict-shaped pytree.

How it works
------------

The exported program already carries pytree specs (``args_spec`` for
``arg_inputs``, ``kwargs_spec`` for ``kwarg_inputs``, ``out_spec`` for the
return value) that fully describe the structure of inputs and outputs.
When you provide binding names as a pytree of strings, Torch-TensorRT
runs ``pytree.tree_flatten`` and compares the resulting ``TreeSpec``
against the exported program's spec.  When they match, the flat list of
names maps 1:1 to FX's flattened placeholder / output order — no runtime
queue, no in-band validation, just an up-front structural check.
"""

import torch
import torch_tensorrt
from torch_tensorrt.dynamo._compiler import BindingNameMismatchError

# %%
import tensorrt as trt

DEVICE = torch.device("cuda", 0)


# %%
# Helpers
# --------
#
# A pair of small helpers: one reads binding names off a deserialized
# engine, the other actually runs the engine via the native TRT Python
# API.  The "execute via native TRT" path is what production deployments
# use — the whole point of this API is that the binding names you supply
# are the names you'll bind by at execution time, not just metadata in
# the engine file.


def deserialize(engine_bytes: bytes) -> trt.ICudaEngine:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(engine_bytes)


def binding_names(engine: trt.ICudaEngine, mode: trt.TensorIOMode) -> list[str]:
    return [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == mode
    ]


_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.BF16: torch.bfloat16,
}


def run_engine(engine: trt.ICudaEngine, named_inputs: dict) -> dict:
    """Execute an engine via the native TRT Python API.

    ``named_inputs`` is a {binding_name: contiguous CUDA tensor} dict.
    Returns {binding_name: output tensor}.  Demonstrates that the
    user-supplied binding names are what production C++/Python TRT
    runtime code will bind by.
    """
    context = engine.create_execution_context()
    for name, tensor in named_inputs.items():
        context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.data_ptr())

    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        dtype = _TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(name)]
        out = torch.empty(shape, dtype=dtype, device=DEVICE)
        context.set_tensor_address(name, out.data_ptr())
        outputs[name] = out

    stream = torch.cuda.Stream(device=DEVICE)
    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


# %%
# Case 1 — positional args, tuple-shaped return
# ----------------------------------------------
#
# Start with the most common shape: ``forward(x)`` returning a 2-tuple.
# ``arg_input_binding_names`` mirrors ``arg_inputs`` (a 1-tuple here);
# ``output_binding_names`` mirrors the return tuple.


class TwoHeads(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.relu(x), torch.tanh(x)


two_heads = TwoHeads().eval().cuda().half()
x = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
exported = torch.export.export(two_heads, (x,))

engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
    exported,
    arg_inputs=(x,),
    arg_input_binding_names=("input_image",),
    output_binding_names=("relu_out", "tanh_out"),
    require_full_compilation=True,
    min_block_size=1,
    use_python_runtime=False,
    immutable_weights=True,
)
engine = deserialize(engine_bytes)
print("Case 1 inputs:", binding_names(engine, trt.TensorIOMode.INPUT))
print("Case 1 outputs:", binding_names(engine, trt.TensorIOMode.OUTPUT))
# Case 1 inputs: ['input_image']
# Case 1 outputs: ['relu_out', 'tanh_out']

# Run the engine through the native TRT API using the names we requested.
trt_outs = run_engine(engine, {"input_image": x.contiguous()})
with torch.no_grad():
    ref_relu, ref_tanh = two_heads(x)
torch.testing.assert_close(trt_outs["relu_out"], ref_relu, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(trt_outs["tanh_out"], ref_tanh, rtol=1e-2, atol=1e-2)
print("Case 1 native TRT run matches PyTorch.")


# %%
# Case 2 — keyword-only inputs
# ------------------------------
#
# When the model takes keyword arguments, you pass ``kwarg_inputs`` and
# match its shape with ``kwarg_input_binding_names``.  Note we leave
# ``arg_input_binding_names`` unset because ``arg_inputs`` is empty.


class KwargOnly(torch.nn.Module):
    def forward(self, image: torch.Tensor, positions: torch.Tensor):
        return image + positions


kwarg_only = KwargOnly().eval().cuda().half()
image = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
positions = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
kw_exported = torch.export.export(
    kwarg_only,
    args=(),
    kwargs={"image": image, "positions": positions},
)

engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
    kw_exported,
    arg_inputs=(),
    kwarg_inputs={"image": image, "positions": positions},
    kwarg_input_binding_names={"image": "rgb_in", "positions": "pos_in"},
    output_binding_names="combined",
    require_full_compilation=True,
    min_block_size=1,
    use_python_runtime=False,
    immutable_weights=True,
)
engine = deserialize(engine_bytes)
print("Case 2 inputs:", sorted(binding_names(engine, trt.TensorIOMode.INPUT)))
print("Case 2 outputs:", binding_names(engine, trt.TensorIOMode.OUTPUT))
# Case 2 inputs: ['pos_in', 'rgb_in']
# Case 2 outputs: ['combined']

trt_outs = run_engine(
    engine,
    {"rgb_in": image.contiguous(), "pos_in": positions.contiguous()},
)
with torch.no_grad():
    ref = kwarg_only(image=image, positions=positions)
torch.testing.assert_close(trt_outs["combined"], ref, rtol=1e-2, atol=1e-2)
print("Case 2 native TRT run matches PyTorch.")


# %%
# Case 3 — nested collections as inputs and outputs
# --------------------------------------------------
#
# Inputs and outputs can be arbitrary nested collections of tensors —
# tuples of dicts of tensors, lists of tuples, anything ``pytree`` can
# flatten.  The binding-name kwargs follow the same nesting.  Here the
# model takes a tuple of two cameras (each a dict of two tensors) and
# returns a dict of feature stacks.


class CameraTower(torch.nn.Module):
    def forward(self, cameras: tuple, bias: torch.Tensor):
        feats = []
        for cam in cameras:
            feats.append(cam["rgb"] + cam["depth"] + bias)
        return {"primary": feats[0], "secondary": feats[1]}


def _cam():
    return {
        "rgb": torch.randn(2, 3, device=DEVICE, dtype=torch.float16),
        "depth": torch.randn(2, 3, device=DEVICE, dtype=torch.float16),
    }


camera_tower = CameraTower().eval().cuda().half()
cameras = (_cam(), _cam())
bias = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
nested_exported = torch.export.export(camera_tower, args=(cameras, bias))

engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
    nested_exported,
    arg_inputs=(cameras, bias),
    arg_input_binding_names=(
        (
            {"rgb": "cam0_rgb", "depth": "cam0_depth"},
            {"rgb": "cam1_rgb", "depth": "cam1_depth"},
        ),
        "global_bias",
    ),
    output_binding_names={"primary": "p_feats", "secondary": "s_feats"},
    require_full_compilation=True,
    min_block_size=1,
    use_python_runtime=False,
    immutable_weights=True,
)
engine = deserialize(engine_bytes)
print("Case 3 inputs:", sorted(binding_names(engine, trt.TensorIOMode.INPUT)))
print("Case 3 outputs:", sorted(binding_names(engine, trt.TensorIOMode.OUTPUT)))
# Case 3 inputs: ['cam0_depth', 'cam0_rgb', 'cam1_depth', 'cam1_rgb', 'global_bias']
# Case 3 outputs: ['p_feats', 's_feats']

trt_outs = run_engine(
    engine,
    {
        "cam0_rgb": cameras[0]["rgb"].contiguous(),
        "cam0_depth": cameras[0]["depth"].contiguous(),
        "cam1_rgb": cameras[1]["rgb"].contiguous(),
        "cam1_depth": cameras[1]["depth"].contiguous(),
        "global_bias": bias.contiguous(),
    },
)
with torch.no_grad():
    ref = camera_tower(cameras, bias)
torch.testing.assert_close(trt_outs["p_feats"], ref["primary"], rtol=1e-2, atol=1e-2)
torch.testing.assert_close(trt_outs["s_feats"], ref["secondary"], rtol=1e-2, atol=1e-2)
print("Case 3 native TRT run matches PyTorch.")


# %%
# Case 4 — structural validation
# -------------------------------
#
# If the shape of any of the binding-name kwargs doesn't match the
# exported program's spec, the converter raises
# ``BindingNameMismatchError`` before any TensorRT network construction.
# The error message shows the expected structure plus a leaf-position
# listing — you can read the correct shape off the error and re-run.

try:
    torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        arg_inputs=(x,),
        output_binding_names=("only_one",),  # wrong arity for the 2-tuple return
        require_full_compilation=True,
        min_block_size=1,
        use_python_runtime=False,
        immutable_weights=True,
    )
except BindingNameMismatchError as err:
    print("Caught BindingNameMismatchError as expected.")
    print(str(err).splitlines()[0])


# %%
# Notes
# -----
#
# * The binding-name kwargs are *parallel* to the input kwargs they refer
#   to: ``arg_input_binding_names`` matches ``arg_inputs``,
#   ``kwarg_input_binding_names`` matches ``kwarg_inputs``.  Skip either
#   one if the corresponding input slot is empty.
# * Duplicate names within any individual list, or across inputs and
#   outputs, are rejected at the API boundary — TensorRT requires
#   binding names to be globally unique.
# * This API is **only** available on
#   ``convert_exported_program_to_serialized_trt_engine``.  ``compile()``
#   and ``dynamo.compile()`` produce ``TorchTensorRTModule`` artifacts
#   whose runtime depends on the default naming policy, so they
#   intentionally don't expose this knob.
