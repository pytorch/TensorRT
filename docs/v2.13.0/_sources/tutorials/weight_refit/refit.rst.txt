.. _refit:

Refitting TensorRT Engines with Updated Weights
================================================

Compiling a TensorRT engine is expensive — it can take seconds to minutes depending on
model size and optimization level. For workflows where the **graph structure is fixed but
the weights change** (fine-tuning checkpoints, LoRA adapters, EMA weight updates),
:func:`~torch_tensorrt.dynamo.refit_module_weights` updates engine weights in-place
without rerunning the TRT optimizer.

----

When to Use Refit
-----------------

Refit is the right tool when:

* You have a compiled TRT module and a new PyTorch checkpoint with the **same
  architecture** but different weights.
* The weight update is frequent (e.g., loading a new LoRA adapter for each request).
* Full recompilation is too slow for your workflow.

Refit is **not** appropriate when:

* The model architecture has changed (different layers, shapes, or graph topology).
  In that case you must recompile. :class:`~torch_tensorrt.MutableTorchTensorRTModule`
  handles this case automatically.
* The compiled engine was built with ``immutable_weights=True`` (the default). You must
  compile with ``immutable_weights=False`` to make engines refittable.

----

Requirements
------------

1. Compile the original model with ``immutable_weights=False``:

   .. code-block:: python

       import torch
       import torch_tensorrt

       model = MyModel().eval().cuda()
       inputs = [torch.randn(1, 3, 224, 224).cuda()]

       exp_program = torch.export.export(model, tuple(inputs))
       compiled = torch_tensorrt.dynamo.compile(
           exp_program,
           arg_inputs=inputs,
           immutable_weights=False,   # required for refit
       )

2. Export the **updated** model (same architecture, new weights) as an
   ``ExportedProgram``:

   .. code-block:: python

       updated_model = MyModel().eval().cuda()
       # load updated_model weights from checkpoint...
       new_exp_program = torch.export.export(updated_model, tuple(inputs))

3. Call :func:`~torch_tensorrt.dynamo.refit_module_weights`:

   .. code-block:: python

       from torch_tensorrt.dynamo import refit_module_weights

       refitted = refit_module_weights(
           compiled_module=compiled,
           new_weight_module=new_exp_program,
       )

   ``refitted`` is a new ``torch.fx.GraphModule`` with the TRT engines updated to use
   the new weights. The original ``compiled`` module is unchanged (a deep copy is made
   by default).

----

API
---

.. code-block:: python

    torch_tensorrt.dynamo.refit_module_weights(
        compiled_module,
        new_weight_module,
        arg_inputs=None,
        kwarg_inputs=None,
        verify_output=False,
        use_weight_map_cache=True,
        in_place=False,
    )

**Parameters**

``compiled_module`` (``torch.fx.GraphModule | ExportedProgram``)
    The compiled TRT module to update. Must have been compiled with
    ``immutable_weights=False``. Can be loaded from disk via
    ``torch_tensorrt.load()``.

``new_weight_module`` (``ExportedProgram``)
    Exported program containing the updated weights. Must have the same model
    architecture (graph topology and tensor shapes) as the original.

``arg_inputs`` (``Tuple[Any, ...]``, optional)
    Sample positional inputs. Required only when ``verify_output=True``.

``kwarg_inputs`` (``dict[str, Any]``, optional)
    Sample keyword inputs. Required only when ``verify_output=True``.

``verify_output`` (``bool``, default ``False``)
    Run a numerical check comparing the output of the refitted TRT engine against
    PyTorch on the provided sample inputs. Useful for catching silent refit failures
    during development.

``use_weight_map_cache`` (``bool``, default ``True``)
    When torch-tensorrt programs are compiled, the TRTIntpereter builds a map of which
    exported program nodes correspond to which TensorRT layers. This mapping is stored as metadata in serialized
    torch-tensorrt programs. This cache is not gaurenteed to be an exact match but to a new
    unseen exported program but when it does, it reduces refit time by ~50%.

``in_place`` (``bool``, default ``False``)
    If ``True``, modify the compiled module in-place rather than returning a copy.
    Not supported for ``ExportedProgram`` inputs (use the returned module instead).

**Returns** ``torch.fx.GraphModule`` — the refitted compiled module.

----

Output Verification
-------------------

Use ``verify_output=True`` during development to catch numerical mismatches between the
refitted TRT engine and PyTorch:

.. code-block:: python

    inputs = [torch.randn(1, 3, 224, 224).cuda()]

    refitted = refit_module_weights(
        compiled_module=compiled,
        new_weight_module=new_exp_program,
        arg_inputs=tuple(inputs),
        verify_output=True,
    )

A warning is logged if the outputs differ beyond floating-point tolerance.

----

Batch Norm and Constant Folding
---------------------------------

BatchNorm layers are typically constant-folded into the preceding convolution during
export. ``refit_module_weights`` handles this automatically: it reconstructs the
folded ``weight``, ``bias``, ``running_mean``, and ``running_var`` tensors from the
updated BatchNorm state dict and maps them to the correct fused TRT layer.

----

Saving and Loading Refitted Modules
-------------------------------------

Refitted modules can be saved and loaded exactly like any other compiled module:

.. code-block:: python

    torch_tensorrt.save(refitted, "model_v2.ep", arg_inputs=inputs)
    # later:
    refitted = torch_tensorrt.load("model_v2.ep")

----

Refit vs. MutableTorchTensorRTModule
--------------------------------------

Use :class:`~torch_tensorrt.MutableTorchTensorRTModule` when you need automatic
handling of both weight mutations and structural mutations:

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Scenario
     - ``refit_module_weights``
     - ``MutableTorchTensorRTModule``
   * - New checkpoint, same architecture
     - Yes — explicit, controlled
     - Yes — automatic
   * - LoRA adapter changes graph topology
     - No — must recompile manually
     - Yes — detects structural change, recompiles automatically
   * - HuggingFace ``diffusers`` integration
     - Requires custom glue code
     - Drop-in ``nn.Module`` replacement
   * - Fine-grained control over refit timing
     - Yes
     - No — mutation-triggered
