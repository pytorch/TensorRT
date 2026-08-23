.. _weight_refit_design:

Weight Refit
=============

.. note::

   This page documents the design for weight refit in Torch-TensorRT.
   Original design discussions:
   `RFC #2900 <https://github.com/pytorch/TensorRT/discussions/2900>`_,
   `RFC #3204 <https://github.com/pytorch/TensorRT/discussions/3204>`_.

Goal
----

Allow compiled TensorRT engines to have their weights updated *after* compilation
without rebuilding the engine from scratch. Engine builds involve expensive kernel
auto-tuning; refit skips that entirely and just copies new weight values into the
already-built engine, which is typically 80–95% faster than a full rebuild.

Primary use cases:

* **LoRA / adapter hot-swapping** — apply a new adapter (e.g. a LoRA for Stable
  Diffusion) to a pre-compiled TRT engine in seconds rather than minutes.
* **A/B testing** — switch model weight variants without recompilation.
* **Cloud pre-compiled engines** — distribute a weight-stripped engine; end users
  fill weights locally.
* **Parameter-efficient fine-tuning** — freeze the backbone in TRT and only refit
  adapter layers on each training step.

High-Level Design
------------------

.. image:: https://github.com/pytorch/TensorRT/assets/123616592/be69b5f9-3266-4024-bd59-0b8f822e608f
   :alt: Weight refit high-level pipeline

The compilation pipeline is extended from::

    lowering → partitioning → compilation

to::

    lowering → partitioning → compilation → refit

During the initial compilation a **refit map** is constructed — a lookup table
mapping original PyTorch parameter names to their corresponding TensorRT layer
indices. This map is stored inside every ``TorchTRTModule`` and is used later to
efficiently copy new weights without traversing the full graph again.

Compilation Modes
------------------

Three engine modes are supported (controlled by ``make_refittable`` and
``strip_engine_weights``):

1. **Weightless + refittable** (``strip_engine_weights=True``,
   ``make_refittable=True``) — engine stores only the computation graph;
   weights are supplied at runtime via refit. Cache-friendly: the engine file
   is much smaller, and any engine is cacheable regardless of weight values.

2. **Refittable with embedded weights** (``make_refittable=True``) — engine
   stores both the computation graph and current weights. Refit replaces the
   embedded weights in-place (``kREFIT_IDENTICAL`` semantic in TensorRT).

3. **Non-refittable** (legacy default) — weights are baked into the engine at
   build time; no post-build updates are possible.

User-Facing API
----------------

``refit_module_weights``
^^^^^^^^^^^^^^^^^^^^^^^^^

Refit a compiled ``torch.fx.GraphModule`` with weights from a new exported program:

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt.dynamo import refit_module_weights

    # Compile once
    exp_program = torch.export.export(model, inputs)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs=inputs)

    # Later: update weights (e.g. different LoRA applied to model)
    new_model = MyModel()   # same architecture, different weights
    new_exp_program = torch.export.export(new_model, inputs)
    refitted_gm = refit_module_weights(
        compiled_module=trt_gm,
        new_weight_module=new_exp_program,
        inputs=inputs,
    )

``MutableTorchTensorRTModule``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A higher-level ``nn.Module`` wrapper that intercepts weight mutations and
dispatches to refit automatically:

.. code-block:: python

    from torch_tensorrt.dynamo import MutableTorchTensorRTModule

    mutable = MutableTorchTensorRTModule(model, config=settings)

    # Weight update (e.g. HuggingFace diffusers LoRA load_lora_weights)
    pipeline.unet = mutable
    pipeline.load_lora_weights("path/to/lora")
    # → intercepted; refit triggered automatically, no recompilation

    # If the model architecture changes (new adapter inserts layers),
    # a full recompilation is triggered instead (engine cache is consulted first).

Internal Implementation
------------------------

Refit Map Construction
^^^^^^^^^^^^^^^^^^^^^^^

During conversion the FX interpreter inspects each ``INetworkDefinition`` layer that
carries learnable weights (convolutions, deconvolutions, BatchNorm, LayerNorm,
constant layers) and records a mapping::

    { "pytorch.param.name" : trt_layer_index }

This map is serialized alongside the engine bytes and stored as part of the
``torch.classes.tensorrt.Engine`` object.

.. code-block:: python

    def construct_refit_mapping(
        module: torch.fx.GraphModule,
        inputs: Sequence[Input],
        settings: CompilationSettings = CompilationSettings(),
    ) -> dict[str, np.ndarray]:
        """
        Run the interpreter and find the weight mapping between
        the exported program's state_dict and TensorRT engine weights.
        Returns: { trt_weight_name -> numpy weight array }
        """

Weight Application
^^^^^^^^^^^^^^^^^^^

``refit_module_weights`` re-runs the compilation settings stored in the compiled
module to re-trace the new exported program through the ATen lowering stage only
(no partitioning or engine rebuild), then:

1. Iterates over TRT submodules in the compiled graph.
2. For each, constructs a fresh ``INetworkDefinition`` from the new weights.
3. Uses ``nvinfer1::IRefitter`` to push the new weights into the existing engine.
4. Returns a copy of the compiled module with the updated engines.

Non-Refittable Ops
^^^^^^^^^^^^^^^^^^^

Some ops cannot be refitted because TensorRT embeds their outputs as constants
(e.g. ``aten.cumsum``, ``aten.embedding_bag``). Two options are available:

* Keep engines refittable and fall back those ops to PyTorch.
* Set ``make_refittable=False`` and rebuild the engine when weights change.

Refit Caching Shortcut
^^^^^^^^^^^^^^^^^^^^^^^

If the mapping between ``state_dict`` keys and TRT engine weight names is stable
across calls (same model, different weights), the map is cached so that
re-interpretation of the exported program can be skipped entirely on subsequent
refits — only the weight copy step runs.

Related
-------

* :ref:`engine_caching_design` — weight refit is invoked automatically on cache hits.
* :ref:`execution` — ``MutableTorchTensorRTModule`` wraps the runtime module.
* `Example: refit_engine_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/refit_engine_example.py>`_
* `Example: mutable_torchtrt_module_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/mutable_torchtrt_module_example.py>`_
