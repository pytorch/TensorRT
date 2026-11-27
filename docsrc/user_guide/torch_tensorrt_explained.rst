.. _torch_tensorrt_explained:

Torch-TensorRT Explained
=================================

Torch-TensorRT is a compiler for PyTorch models targeting NVIDIA GPUs
via the TensorRT Model Optimization SDK. It aims to provide better
inference performance for PyTorch models while still maintaining the
great ergonomics of PyTorch.

Dynamo Frontend
-----------------

The Dynamo frontend is the default frontend for Torch-TensorRT. It utilizes the `dynamo compiler stack <https://pytorch.org/docs/stable/torch.compiler_deepdive.html>`_ from PyTorch.


``torch.compile`` (Just-in-time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch.compile`` is a JIT compiler stack, as such, compilation is deferred until first use. This means that as conditions change in the graph, the graph will automatically recompile.
This provides users the most runtime flexibility, however limits options regarding serialization.

Under the hood, `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile>`_ delegates subgraphs it believes can be lowered to Torch-TensorRT. Torch-TensorRT further lowers these graphs into ops consisting of solely `Core ATen Operators <https://pytorch.org/executorch/stable/ir-ops-set-definition.html>`_
or select "High-level Ops" amenable to TensorRT acceleration. Subgraphs are further partitioned into components that will run in PyTorch and ones to be further compiled to TensorRT based
on support for operators. TensorRT engines then replace supported blocks and a hybrid subgraph is returned to ``torch.compile`` to be run on call.

Accepted Formats
...................
- torch.fx GraphModule (``torch.fx.GraphModule``)
- PyTorch Module (``torch.nn.Module``)

Returns
...................
- Boxed-function that triggers compilation on first call


``torch_tensorrt.dynamo.compile`` (Ahead-of-time)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch_tensorrt.dynamo.compile`` is an AOT compiler, models are compiled in an explicit compilation phase. These compilation artifacts can then be serialized and reloaded at a later date.
Graphs go through the ``torch.export.trace`` system to be lowered into a graph consisting of `Core ATen Operators <https://pytorch.org/executorch/stable/ir-ops-set-definition.html>`_ or select "High-level Ops" amenable to TensoRT acceleration.
Subgraphs are further partitioned into components that will run in PyTorch and ones to be further compiled to TensorRT based on support for operators. TensorRT engines then replace supported blocks
and a hybrid subgraph is packed into an `ExportedProgram <https://pytorch.org/docs/stable/export.ir_spec.html#exportedprogram>`_ which can be serialized and reloaded.

Accepted Formats
...................
- torch.export.ExportedProgram (``torch.export.ExportedProgram``)
- torch.fx GraphModule (``torch.fx.GraphModule``) (via ``torch.export.export``)
- PyTorch Module (``torch.nn.Module``) (via ``torch.export.export``)

Returns
...................
- torch.fx.GraphModule (serializable with ``torch.export.ExportedProgram``)

Legacy Frontends
------------------

TorchScript and the legacy FX frontend have been removed. Use the Dynamo frontend
with ``torch.compile`` or ``torch.export``, and export with AOTInductor or ExecuTorch.
Dynamo continues to accept and return ``torch.fx.GraphModule`` objects.

``torch_tensorrt.compile``
----------------------------------

As there are many different frontends and supported formats, we provide a convenience layer called ``torch_tensorrt.compile`` which lets users access
all the different compiler options. You can specify to ``torch_tensorrt.compile`` what compiler path to use by setting the ``ir`` option, telling
Torch-TensorRT to try to lower the provided model through a specific intermediate representation.

``ir`` Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``torch_compile``: Use the ``torch.compile`` system. Immediately returns a boxed-function that will compile on first call
- ``dynamo``: Run the graph through the ``torch.export``/ torchdynamo stack. If the input module is a ``torch.nn.Module``, it must be "export-traceable" as the module will be traced with ``torch.export.export``. Returns a ``torch.fx.GraphModule`` which can be run immediately or saved via ``torch.export.export`` or ``torch_tensorrt.save``
