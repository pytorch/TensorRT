Torch-TensorRT Edge-LLM
======================

This is an out-of-tree package built on Torch-TensorRT. It owns Edge exporters,
the ``torch.ops.tensorrt_edge_llm.*`` operator namespace, and the Edge-LLM
ExecuTorch delegate. It can be extracted from ``tools/hf`` without changing
Torch-TensorRT's Python package or native backend.

Install alongside a compatible Torch-TensorRT build::

    pip install ./tools/hf

Model exporter dependencies and ExecuTorch are optional extras. Operator
registration does not import model families or ExecuTorch::

    pip install './tools/hf[executorch,exporters]'

Export and save
---------------

Exporters insert named runtime operators into an ``ExportedProgram``. Embedded
operators carry engine bytes as persistent buffers and versioned metadata as
operator arguments. Python execution reconstructs the TensorRT runtime from
these values, including after ``torch.export.save`` and ``torch.export.load``.
The process-local engine cache is only an execution optimization.

The first embedded component is ``vision_tower`` for the FP16 HWC VitRunner
interface, with one input and one output. Existing exporter composition uses
``execute_engine``, ``fuse_prefix``, and ``scatter_image_tokens`` in the same
namespace. ``execute_engine`` still refers to engine directories; it is not an
embedded native module contract. Language/cache and action interfaces need their
own state and aliasing contracts before native delegation is enabled for them.

For an existing vision engine directory containing ``config.json`` and the
serialized engine:

.. code-block:: python

    import torch
    import torch_tensorrt
    import torch_tensorrt_edge_llm  # registers the opset
    from torch_tensorrt_edge_llm import EdgeLLMPartitioner
    from torch_tensorrt_edge_llm.artifact import build_vision_artifact
    from torch_tensorrt_edge_llm.vision import export_vision

    artifact = build_vision_artifact("edge_engines/vision")
    pixels = torch.randn(1, 224, 224, 3, device="cuda", dtype=torch.float16)
    program = export_vision(artifact, pixels)
    result = program.module()(pixels)
    torch.export.save(program, "vision.pt2")

    torch_tensorrt.save(
        program,
        "vision.pte",
        output_format="executorch",
        partitioners=[EdgeLLMPartitioner()],
    )

``partitioners=`` is Torch-TensorRT's public extension point. Each supported
runtime operator becomes an ``EdgeLLMBackend`` delegate. The backend serializes
an EL01 envelope containing the operator metadata and Torch-TensorRT's TR01/TR02
engine blob. The native adapter and Python implementation consume the same
embedded engine; the saved program does not refer to an exporter-side registry.
Additional partitioners may be passed for the surrounding graph.

Build compiled artifacts with Bazel
----------------------------------

Run Bazel from this project's root (``tools/hf`` in the current checkout).
Use Torch-TensorRT as an external Bazel module, and supply the companion
TensorRT-Edge-LLM adapter headers and shared library::

    bazel test --override_module=torch_tensorrt=/path/to/Torch-TensorRT \
        //cpp:test_blob_header
    bazel build --override_module=torch_tensorrt=/path/to/Torch-TensorRT \
        --repo_env=EDGELLM_INCLUDE_DIR=/path/to/TensorRT-Edge-LLM/cpp \
        --repo_env=EDGELLM_EXECUTORCH_LIBRARY=/path/to/libedgellmExecutorch.so \
        //:executorch_backend_archive

The resulting ``bazel-bin/libexecutorch_edge_llm_backend.a`` includes the native
Edge delegate and EL01 parser. An application must also link the compatible
Torch-TensorRT native backend, ExecuTorch, CUDA, TensorRT, and Edge-LLM adapter.
The companion adapter must expose ``executorch/vitExecutorchAdapter.h``; this
interface is required independently of the Python package.

Application source builds with CMake
-----------------------------------

Add ExecuTorch and Torch-TensorRT's ExecuTorch backend to the application first,
then add this package's ``cpp`` directory. Link
``tensorrt_edge_llm::executorch_backend``. That target preserves static delegate
registration and shares Torch-TensorRT's ``extension_cuda`` caller-stream
library. The Edge-LLM headers and library are supplied using
``EDGELLM_INCLUDE_DIR`` and ``EDGELLM_EXECUTORCH_LIBRARY``.

A complete source-build example lives in ``examples/executorch_reference_runner``::

    cmake -S examples/executorch_reference_runner -B build/runner \
        -DEXECUTORCH_SOURCE_DIR=/path/to/executorch \
        -DTORCHTRT_EXECUTORCH_SOURCE_DIR=/path/to/Torch-TensorRT/cpp/src/torch_tensorrt/executorch \
        -DEDGELLM_INCLUDE_DIR=/path/to/TensorRT-Edge-LLM/cpp \
        -DEDGELLM_EXECUTORCH_LIBRARY=/path/to/libedgellmExecutorch.so
    cmake --build build/runner
    build/runner/edge_llm_executorch_runner --model_path=vision.pte

Validation
----------

Run ``python -m pytest tests`` from this project. Operator registration and
Python export tests do not require ExecuTorch. Lowering tests use the optional
ExecuTorch extra. The native parser test does not require the companion SDK.
