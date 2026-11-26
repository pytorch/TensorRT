.. _execution:

Runtime Phase
================

The runtime phase wraps the compiled TensorRT engines together with any remaining
PyTorch subgraphs into a single callable module and provides the execution
infrastructure for inference.

Dynamo Runtime (Primary Path)
-------------------------------

The Dynamo runtime is fully serializable and supports advanced features like
CUDAGraphs and multi-device safety.

TensorRT engines are stored as ``torch.classes.tensorrt.Engine`` — a C++ TorchBind
class that holds the serialized engine bytes plus metadata:

* Engine name
* Refit map (PyTorch parameter name → TensorRT layer index)
* Function signature (input/output names, dtypes, shapes)
* Runtime requirements (e.g. whether an output allocator is needed for DDS ops)
* Target TensorRT version and hardware compatibility flags

Inference is triggered via the ``torch.ops.tensorrt.execute_engine`` custom op:

.. code-block:: text

    tensorrt::execute_engine(
        Tensor[] input_tensors,
        __torch__.torch.classes.tensorrt.Engine engine
    ) -> Tensor[]

This op pops inputs and the engine off the PyTorch dispatcher stack, runs the tensors
through TensorRT, and pushes output tensors back. The compiled ``torch.fx.Graph``
stores engine objects as attributes, making the whole module portable.

Serialization Options
----------------------

`ExportedProgram <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html>`_ (``torch.export``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default serialization path for the Dynamo AOT workflow. The compiled
``torch.fx.GraphModule`` is wrapped in a
`torch.export.ExportedProgram <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html>`_
container. TensorRT engines are stored as tensor attributes in the package; PyTrees
capture input/output structure.

.. code-block:: python

    torch_tensorrt.save(trt_gm, "model.ep", arg_inputs=inputs)
    # later:
    trt_gm = torch_tensorrt.load("model.ep")

AOTInductor (``torch._export.aot_compile``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorRT engines can be baked into AOTInductor-generated shared objects alongside
TorchInductor-compiled kernels. Any PyTorch-backed subgraphs become Inductor-generated
Triton kernels. The result is deployable without Python — only
``libtorchtrt_runtime.so`` is needed at runtime.

See ``examples/torchtrt_aoti_example`` for a full example.

Stand-alone TensorRT Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Individual TensorRT engines can also be extracted and run standalone with ``trtexec``
or any other TensorRT-compatible runtime, entirely outside of PyTorch.

MutableTorchTensorRTModule
---------------------------

``MutableTorchTensorRTModule`` is a higher-level wrapper for use cases that require
weight mutability (e.g. LoRA adapters on diffusion models).

It maintains two graphs in parallel: the original PyTorch
`nn.Module <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
and the compiled TRT graph. User interactions — including weight assignments via
standard PyTorch APIs or HuggingFace ``diffusers`` — hit the PyTorch graph as normal. The
module intercepts these and:

* **Weight mutations** (same graph structure, different weights) → triggers a fast
  refit using the refit map constructed during conversion. No recompilation needed.
* **Structural mutations** (e.g. a new LoRA adapter changes the graph topology) →
  triggers a full recompilation, using the engine cache to skip unchanged subgraphs.

This gives the ergonomics of a regular ``nn.Module`` with TensorRT performance, and is
compatible with HuggingFace ``diffusers`` LoRA workflows without any code changes.

----
