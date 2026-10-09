.. _export_serialization_design:

Export and Serialization
=========================

.. note::

   This page documents the design for serialization of Torch-TensorRT compiled
   programs.
   Original design discussion:
   `RFC #2176 <https://github.com/pytorch/TensorRT/discussions/2176>`_.

Goal
----

Allow a compiled Torch-TensorRT program to be saved to disk and loaded back
without recompilation. The loaded program must be executable on any compatible
device without importing any other model weights separately.

.. image:: https://github.com/pytorch/TensorRT/assets/10511428/a25610c1-74ee-4c9f-bcac-078e66a74c98
   :alt: Export serialization workflow

Serialization Formats
----------------------

Two formats are supported:

``torch.export`` / ``ExportedProgram`` (``.ep``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default format for the ``torch.export`` (AOT) workflow. The compiled
``torch.fx.GraphModule`` is wrapped in a
``torch.export.ExportedProgram`` container.

How TRT engines are stored:

* Each compiled TRT subgraph uses the ``torch.ops.tensorrt.execute_engine``
  custom op as a ``call_function`` node in the FX graph. This is serializable
  by the standard ``torch.export`` serialization stack.
* TRT engine bytes are serialized as tensor attributes in the ``ExportedProgram``
  package. Input/output structures are captured as PyTrees.

.. code-block:: python

    # Save
    torch_tensorrt.save(trt_gm, "model.ep", arg_inputs=inputs)

    # Load (no recompilation)
    trt_gm = torch_tensorrt.load("model.ep")
    output = trt_gm(*inputs)

.. note::

   The C++ runtime is required for ``ExportedProgram`` serialization. The Python
   runtime does not support this format.

AOTInductor (``.so``)
^^^^^^^^^^^^^^^^^^^^^^

TRT engines can be embedded into an AOTInductor-generated shared library alongside
TorchInductor-compiled Triton kernels. The result is deployable without Python —
only ``libtorchtrt_runtime.so`` is needed at runtime:

.. code-block:: python

    torch._export.aot_compile(
        trt_gm,
        args=inputs,
        options={"aot_inductor.output_path": "model.so"},
    )

    # Runtime (no Python, no PyTorch)
    # load with libtorchtrt_runtime.so only

Stand-Alone TRT Engine (``.trt`` / ``.engine``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Individual TRT engines can be extracted and serialized without any PyTorch wrapper:

.. code-block:: python

    trt_engine_bytes = torch_tensorrt.convert_exported_program_to_trt_engine(
        exp_program, inputs=inputs
    )
    with open("engine.trt", "wb") as f:
        f.write(trt_engine_bytes)

These can be run with ``trtexec`` or any other TensorRT-compatible runtime.

Internal Design
----------------

The key constraint is that ``call_module`` nodes (submodule calls) are **not**
serializable by the standard ``torch.export`` serializer. Torch-TensorRT solves
this by embedding TRT engines as ``call_function`` nodes using the custom
``torch.ops.tensorrt.execute_engine`` operator, which *is* serializable:

.. code-block:: text

    # call_function node — serializable
    %execute_engine = call_function[
        target=torch.ops.tensorrt.execute_engine
    ](args=([%arg7_1], <TRT engine bytes>), kwargs={})

Engine bytes are stored as opaque constant attributes attached to the graph and
packed into the ``ExportedProgram`` zip archive alongside the model weights.

Custom serializers (``TorchTRTExportedProgramSerializer``,
``TorchTRTSerializer``) handle the ``execute_engine`` node type during
``torch.export`` serialization. Corresponding deserializers reconstruct the engine
from bytes and restore the ``call_function`` node.

Versioning
^^^^^^^^^^^

Serialized programs include the Torch-TensorRT version, TensorRT version, and
target device SM capability. A compatibility check at load time warns if the
serialized engine was built for a different device or library version.

Related
-------

* :ref:`execution` — the runtime that executes loaded programs.
* :ref:`engine_caching_design` — engine caching uses a different (faster) on-disk
  format optimized for repeated compilations rather than deployment.
* `Example: save_dynamic_shapes_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/save_dynamic_shapes_example.py>`_
