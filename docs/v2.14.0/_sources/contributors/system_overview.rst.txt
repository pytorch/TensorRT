.. _system_overview:

System Overview
================

Goal
----

Torch-TensorRT's goal is to allow users of PyTorch to access the performance of TensorRT
using familiar PyTorch workflows and APIs — without requiring them to leave the PyTorch
ecosystem or learn a separate optimization stack.

The primary frontend is the **Dynamo** path, which integrates with
`torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html>`_ and
`torch.export <https://pytorch.org/docs/stable/export.html>`_. A legacy **TorchScript**
(``ts``) path is also supported for backwards compatibility.

Repository Structure
---------------------

* ``py/``: Python API and the Dynamo compiler pipeline (lowering, partitioning, conversion, runtime)
* ``core/``: C++ runtime library — TensorRT engine management, the ``execute_engine`` custom op, serialization
* ``cpp/``: C++ API surface
* ``tests/``: Test suite. Python tests under ``tests/py/``, C++ tests under ``tests/core/``
* ``tools/``: Developer utilities (opset coverage, perf benchmarking, LLM tools)
* ``examples/``: Standalone example programs
* ``docsrc/``: Documentation source (RST)
* ``third_party/``: Build dependency declarations

High-Level Architecture
------------------------

The Dynamo compiler pipeline has five stages:

.. code-block:: text

                    PyTorch Model
                        │
                        ▼
    torch.compile  ─── or ───  torch.export
        │                           │
        └──────────┬────────────────┘
                   ▼
           TorchDynamo / FX Graph
                   │
                   ▼
              Lowering passes
           (Core ATen decompositions
            + subgraph rewriting)
                   │
                   ▼
             Partitioning
         (TensorRT vs PyTorch subgraphs)
                   │
                   ▼
         Conversion (per TRT subgraph)
       FX interpreter → INetworkDefinition
                   │
                   ▼
          Module Wrapping + Runtime
         TorchTRTModule (C++ or Python)

Two Entry Points
-----------------

``torch.compile`` (JIT)
^^^^^^^^^^^^^^^^^^^^^^^

Hooks into PyTorch's JIT compilation system. Dynamo captures subgraphs lazily at
runtime using guards — if inputs violate a guard the subgraph is recompiled.
Compilation artifacts are cached but not directly serializable by the user; the
backend (Torch-TensorRT) is responsible for caching compiled engines.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    x = torch.randn((1, 3, 224, 224)).cuda()

    optimized_model = torch.compile(model, backend="tensorrt")
    optimized_model(x)  # compiled on first call, fast thereafter

``torch.export`` + ``torch_tensorrt.dynamo.compile`` (AOT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traces the model once ahead-of-time via ``torch.export``. Because there is no
runtime recompilation the user must provide explicit dynamic-shape annotations
upfront. The result is an ``ExportedProgram`` that is serializable.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]

    trt_gm = torch_tensorrt.compile(model, arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ep", arg_inputs=inputs)

How TorchDynamo Works
----------------------

Dynamo hooks into CPython's frame evaluation API
(`PEP 523 <https://peps.python.org/pep-0523/>`_) and dynamically rewrites Python
bytecode just before execution to extract sequences of PyTorch operations into
`torch.fx.Graph <https://docs.pytorch.org/docs/stable/fx.html>`_ objects. These
graphs are handed off to the configured backend (Torch-TensorRT) for compilation.

For a thorough treatment of how Dynamo works internally, see the
`TorchDynamo Overview <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_overview.html>`_
and the
`Dynamo Deep-Dive <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html>`_.

Key properties of captured graphs:

* **FX Graphs** — simple node-by-node graph IR. Each ``torch.fx.Node`` wraps one
  operator call with typed inputs and outputs.
  See `torch.fx <https://docs.pytorch.org/docs/stable/fx.html>`_.
* **Core ATen opset** — all high-level PyTorch ops decompose to the ~250-op
  `Core ATen IR <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_ir.html>`_,
  which is purely functional and carries guaranteed shape/dtype metadata.
* **Symbolic shapes (SymPy)** — dynamic dimensions are represented as symbolic
  integer expressions (e.g. ``(2 * s0, 3)``) throughout the graph, enabling
  downstream stages to compute valid shape ranges for TensorRT without re-running
  the model. See
  `Symbolic Shapes <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html#symbolic-shapes>`_.
* **FakeTensor mode** — shape and dtype propagation is done using
  `FakeTensor <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_fake_tensor.html>`_
  objects (tensors that carry metadata but no real data). This replaces the old
  approach of running the model with real data at compile time.
* **Guards** — graph validity conditions checked at runtime. A guard failure triggers
  recompilation. Guards can encode properties like input shape ranges. See
  `What is a guard? <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_overview.html#what-is-a-guard>`_.
