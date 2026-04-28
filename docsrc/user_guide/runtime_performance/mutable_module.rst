.. _mutable_module:

MutableTorchTensorRTModule
===========================

``MutableTorchTensorRTModule`` is a drop-in wrapper for any ``torch.nn.Module`` that
compiles to TRT on the first forward call and **automatically refits the TRT engine
whenever weights change** — without recompilation. It is the recommended path for
integrating Torch-TensorRT into diffusion pipelines (Stable Diffusion, FLUX, etc.) and
any workflow that swaps LoRA adapters or checkpoint weights between runs.

See the complete worked example: :ref:`mutable_torchtrt_module_example`

----

Diffusers / LoRA Use Case
--------------------------

The primary use case is replacing a sub-module inside a HuggingFace
``DiffusionPipeline``. This is a one-line change — the rest of your pipeline code
stays identical:

.. code-block:: python

    import torch
    import torch_tensorrt
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")

    # The only extra line you need
    pipe.unet = torch_tensorrt.MutableTorchTensorRTModule(
        pipe.unet,
        use_python_runtime=True,
    )

The pipeline's ``unet`` is now backed by a TRT engine. The first call to ``pipe(...)``
triggers compilation; subsequent calls run the cached engine.

**Dynamic shapes** are required for diffusion pipelines because batch size and image
dimensions vary. Set the ranges before the first call:

.. code-block:: python

    BATCH = torch.export.Dim("BATCH", min=2, max=24)
    _H    = torch.export.Dim("_H",    min=16, max=32)
    _W    = torch.export.Dim("_W",    min=16, max=32)

    pipe.unet.set_expected_dynamic_shape_range(
        args_dynamic_shape=({0: BATCH, 2: 4 * _H, 3: 4 * _W}, {}),
        kwargs_dynamic_shape={
            "encoder_hidden_states": {0: BATCH},
            "added_cond_kwargs": {
                "text_embeds": {0: BATCH},
                "time_ids":    {0: BATCH},
            },
            "return_dict": None,   # None-valued kwargs are excluded
        },
    )

    image = pipe("cinematic photo, 4k", num_inference_steps=30).images[0]
    image.save("without_lora.jpg")

**Loading a LoRA adapter** uses the standard HuggingFace API — no Torch-TensorRT
calls required. ``MutableTorchTensorRTModule`` detects the weight change and refits
automatically on the next forward call:

.. code-block:: python

    pipe.load_lora_weights(
        "stablediffusionapi/load_lora_embeddings",
        weight_name="all-disney-princess-xl-lo.safetensors",
        adapter_name="lora1",
    )
    pipe.set_adapters(["lora1"], adapter_weights=[1])
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # Refit is triggered automatically here — much faster than recompilation
    image = pipe("cinematic photo, princess", num_inference_steps=30).images[0]
    image.save("with_lora.jpg")

----

Basic Workflow
--------------

For a simpler illustration without diffusers, see the ResNet portion of
:ref:`mutable_torchtrt_module_example`:

.. code-block:: python

    import torch
    import torch_tensorrt
    import torchvision.models as models

    model = models.resnet18(pretrained=True).eval().cuda()
    mutable_module = torch_tensorrt.MutableTorchTensorRTModule(model)

    inputs = [torch.rand(1, 3, 224, 224).cuda()]

    with torch.no_grad():
        mutable_module(*inputs)                          # compiles on first call

        model2 = models.resnet18(pretrained=False).eval().cuda()
        mutable_module.load_state_dict(model2.state_dict())  # marks for refit

        output = mutable_module(*inputs)                 # refits, then runs

----

Dynamic Shapes
--------------

The dynamic-shape hint format mirrors ``torch.export.export``'s ``dynamic_shapes``
argument. Use ``{}`` for inputs whose shapes are static, and a ``torch.export.Dim``
for each dynamic axis. Nested dict/list structures (common in diffusion models) are
fully supported — the hint structure must mirror the input structure exactly:

.. code-block:: python

    dim_0 = torch.export.Dim("dim_0", min=1, max=50)
    dim_1 = torch.export.Dim("dim_1", min=1, max=50)

    mutable_module.set_expected_dynamic_shape_range(
        args_dynamic_shape=({0: dim_0}, {1: dim_1}),
        kwargs_dynamic_shape={},
    )

Calling ``set_expected_dynamic_shape_range`` again clears the cached inputs and
triggers a fresh compilation on the next forward pass.

----

Engine Caching
--------------

Pass engine-cache settings directly to ``MutableTorchTensorRTModule`` to avoid
recompiling on subsequent process starts:

.. code-block:: python

    mutable_module = torch_tensorrt.MutableTorchTensorRTModule(
        model,
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache_size=1 << 30,  # 1 GiB
    )

See :ref:`engine_cache` for details.

----

Saving and Loading
------------------

``MutableTorchTensorRTModule`` uses its own save/load API (not ``torch_tensorrt.save``)
because it carries extra state — dynamic-shape descriptors, refit state, etc.:

.. code-block:: python

    # Requires use_python_runtime=False (the default)
    torch_tensorrt.MutableTorchTensorRTModule.save(mutable_module, "module.pkl")
    mutable_module = torch_tensorrt.MutableTorchTensorRTModule.load("module.pkl")

``use_python_runtime=True`` (used in the diffusers examples for pipeline compatibility)
does **not** support save/load. Switch to the default C++ runtime if serialization is
required.

----

How the Refit / Recompile Decision Works
-----------------------------------------

Every attribute write, ``load_state_dict``, and sub-module access is intercepted by
an internal ``ChangeTriggerWrapper``. On each forward call the module evaluates an
internal flag:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - State
     - Action
   * - ``LIVE``
     - Inputs match — run the TRT engine directly.
   * - ``NEEDS_REFIT``
     - Weights changed, structure is the same — refit the engine, then run.
   * - ``NEEDS_RECOMPILE``
     - Structure changed (new keys, weight shapes) or input shapes changed —
       full recompile, then run.
   * - ``UNKNOWN``
     - A ``Parameter`` was accessed. Run the original PyTorch model with the
       cached inputs, compare outputs, then decide between ``NEEDS_REFIT`` and
       ``LIVE``.

If refit fails the module falls back to a full recompile automatically.

----

Comparison with ``dynamo.compile``
------------------------------------

For large models like FLUX where a single upfront compilation is preferred and LoRA
swapping is not needed, ``torch_tensorrt.dynamo.compile`` with
``immutable_weights=False`` is the right tool. See :ref:`torch_export_flux_dev` for
the complete example.

.. list-table::
   :widths: 35 32 33
   :header-rows: 1

   * -
     - ``MutableTorchTensorRTModule``
     - ``dynamo.compile`` + manual swap
   * - Weight change detection
     - Automatic
     - Manual
   * - LoRA with diffusers
     - Drop-in (``pipe.unet = ...``)
     - Requires manual refit after LoRA fusion
   * - Save / Load
     - ``save`` / ``load`` (C++ runtime only)
     - ``torch_tensorrt.save`` / ``load``
   * - Dynamic shapes
     - ``set_expected_dynamic_shape_range``
     - ``Input`` min/opt/max
   * - Best for
     - Diffusion pipelines, LoRA swaps, iterative fine-tuning
     - Large one-shot models (FLUX, LLMs), maximum control
