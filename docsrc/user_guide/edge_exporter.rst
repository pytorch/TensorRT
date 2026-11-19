.. _edge_exporter:

Edge Exporter
=============

Export a HuggingFace or LeRobot policy to TensorRT-Edge-LLM engines, then return
one ``torch.export`` graph that calls those engines.

.. code-block:: python

    from hf.exporters import EdgeExporter, EdgeConfig

    exporter = EdgeExporter()
    config = EdgeConfig(dryrun=True, engine_dir="/tmp/pi05_edge")
    exported = exporter.export(policy, {"device": device, "dtype": torch.float16}, config)

``EdgeExporter`` is a HuggingFace ``DynamoExporter``. The public call is the same
shape as Transformers export: ``exporter.export(model, inputs, config)``. The
difference is what happens inside. Instead of tracing the whole policy as one
graph, Edge compiles one TensorRT engine per component, then records a small
outer graph that only *calls* those engines.

The code is in ``tools/hf``. The entry points are ``tools/hf/run_pi05_export.py``,
``run_groot_export.py``, and ``run_nemotron_export.py``.

.. note::

   The Edge exporter is experimental. Family patches target a specific modeling
   surface (LeRobot PI05 / GR00T, HuggingFace Nemotron-H) and TensorRT-Edge-LLM
   plugins. Treat patches as tied to the versions you test against.

.. list-table::
   :header-rows: 1
   :widths: 18 18 40 24

   * - Family
     - Spec key
     - Engines
     - Typical checkpoint
   * - PI05
     - ``pi05``
     - ``vision``, ``language``, ``action``
     - ``lerobot/pi05_libero_base``
   * - GR00T
     - ``groot``
     - ``vision``, ``language``, ``context_projection``, ``action``
     - ``nvidia/GR00T-N1.5-3B``
   * - Nemotron-H
     - ``nemotron_h``
     - ``language``
     - ``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16``

Installation
------------

You need Torch-TensorRT, TensorRT-Edge-LLM plugins, and the modeling stacks for
the families you export.

.. code-block:: bash

    pip install transformers

VLA families (PI05, GR00T) also need LeRobot. Nemotron-H uses Transformers only.

For a real compile, set ``EDGE_LLM_PLUGIN_SO`` (or ``EDGELLM_PLUGIN_PATH`` /
``EDGELLM_TRT_PLUGIN_SO``) to ``libNvInfer_edgellm_plugin.so``.

Before ``export()``, load the Edge-LLM plugins and force HuggingFace attention to
``eager`` (FlashAttention / SDPA are not the plugin path):

.. code-block:: python

    from hf.exporters.plugin.plugin_utils import load_plugins_for_trt
    from hf.exporters.utils import force_hf_attention

    load_plugins_for_trt()
    force_hf_attention(policy.model.paligemma_with_expert.paligemma.model.vision_tower, "eager")
    force_hf_attention(policy.model.paligemma_with_expert.paligemma.model.language_model, "eager")

Export a model
--------------

All families share one interface. Create an exporter, pass a policy or causal LM
plus sample inputs, and call ``export()``.

.. code-block:: python

    from hf.exporters import EdgeExporter, EdgeConfig
    from hf.exporters.plugin.plugin_utils import load_plugins_for_trt

    load_plugins_for_trt()

    exporter = EdgeExporter()
    config = EdgeConfig(
        model_type="pi05",          # optional when the spec can infer it
        engine_dir="/tmp/pi05_edge",
        max_seq_len=968,
        dryrun=True,                # skip TensorRT; still writes config.json + the outer graph
    )
    exported = exporter.export(policy, {"device": device, "dtype": torch.float16}, config)

    # engines: {"vision": ".../vision", "language": ".../language", "action": ".../action"}
    print(exporter.engines)

    # run the exported program
    outputs = exported.module()(**exporter.sample)

Pass the **policy** for PI05 and GR00T (the spec needs the preprocessor), not an
inner submodule. Pass the HuggingFace causal LM for Nemotron.

``dryrun=True`` walks the same export path without building TensorRT engines. Each
component directory still gets a ``config.json``. Use that to debug packing and
patches, then set ``dryrun=False`` (or pass ``--compile`` on the example scripts)
to emit ``.engine`` files.

On a real compile, ``engine_dir/<component>/`` contains ``config.json`` and the
serialized engine (for example ``visual.engine``, ``language.engine``).

Export Program
--------------

``EdgeExporter.export`` returns an ``ExportedProgram``. Each policy
component (vision, language, action, …) is compiled into its own TensorRT
engine under ``engine_dir/<name>/``. The exported graph is the runtime that
calls those engines in order through ``torch.ops.edge_llm.execute_engine``.
A packing op sits between vision and language so image tokens land in the
text embeddings. ``print(program.graph)`` prints that FX graph: each
``execute_engine`` node is one component, and the path in its args is the
engine directory.

Here is a GR00T dryrun (``vision`` → ``scatter_image_tokens`` →
``language`` → ``context_projection`` → ``action``):

.. code-block:: text

    graph():
        %pixel_values : [num_users=1] = placeholder[target=pixel_values]
        %lang_embeds : [num_users=1] = placeholder[target=lang_embeds]
        %image_token_mask : [num_users=1] = placeholder[target=image_token_mask]
        %rope_rotary_cos_sin : [num_users=1] = placeholder[target=rope_rotary_cos_sin]
        %context_lengths : [num_users=1] = placeholder[target=context_lengths]
        %kvcache_start_index : [num_users=1] = placeholder[target=kvcache_start_index]
        %last_token_ids : [num_users=1] = placeholder[target=last_token_ids]
        %ds_stack : [num_users=1] = placeholder[target=ds_stack]
        %past_key_values_0 : [num_users=1] = placeholder[target=past_key_values_0]
        %step_actions : [num_users=1] = placeholder[target=step_actions]
        %step_timestep : [num_users=1] = placeholder[target=step_timestep]
        %state : [num_users=1] = placeholder[target=state]
        %embodiment_id : [num_users=1] = placeholder[target=embodiment_id]
        %execute_engine : [num_users=1] = call_function[target=torch.ops.edge_llm.execute_engine.default](args = (edge_engines/vision, vision, [%pixel_values]), kwargs = {})
        %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%execute_engine, 0), kwargs = {})
        %scatter_image_tokens : [num_users=1] = call_function[target=torch.ops.edge_llm.scatter_image_tokens.default](args = (%getitem, %lang_embeds, %image_token_mask), kwargs = {})
        %execute_engine_1 : [num_users=4] = call_function[target=torch.ops.edge_llm.execute_engine.default](args = (edge_engines/language, language, [%scatter_image_tokens, %rope_rotary_cos_sin, %context_lengths, %kvcache_start_index, %last_token_ids, %ds_stack, %past_key_values_0]), kwargs = {})
        %getitem_1 : [num_users=0] = call_function[target=operator.getitem](args = (%execute_engine_1, 0), kwargs = {})
        %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%execute_engine_1, 1), kwargs = {})
        %getitem_3 : [num_users=0] = call_function[target=operator.getitem](args = (%execute_engine_1, 2), kwargs = {})
        %getitem_4 : [num_users=0] = call_function[target=operator.getitem](args = (%execute_engine_1, 3), kwargs = {})
        %execute_engine_2 : [num_users=1] = call_function[target=torch.ops.edge_llm.execute_engine.default](args = (edge_engines/context_projection, context_projection, [%getitem_2]), kwargs = {})
        %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%execute_engine_2, 0), kwargs = {})
        %execute_engine_3 : [num_users=1] = call_function[target=torch.ops.edge_llm.execute_engine.default](args = (edge_engines/action, action, [%step_actions, %step_timestep, %getitem_5, %state, %embodiment_id]), kwargs = {})
        %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%execute_engine_3, 0), kwargs = {})
        return (getitem_6,)

PI05 is the same shape with ``fuse_prefix`` instead of ``scatter_image_tokens``
and no ``context_projection`` engine. Nemotron is a single
``execute_engine`` on ``language``.

``execute_engine`` is a Torch custom op, not a normal ``nn.Module`` call. Family
``spec.run()`` calls ``call_engine(...)``, so ``torch.export`` records **one
node per engine**. Matching ``register_fake`` kernels give Dynamo the output
shapes. Two packing ops live in the same file
(``tools/hf/exporters/ops.py``):

* ``edge_llm::fuse_prefix`` — PI05: concat vision tokens with language
  embeddings and gather the compact prefix.
* ``edge_llm::scatter_image_tokens`` — GR00T: write vision tokens into the
  ``<image>`` slots of the language embeddings.

These appear in the **outer** ExportedProgram. They are not TensorRT plugins.

Patches
-------

Edge does not wrap the policy in a new module. It temporarily replaces
``Class.forward`` on the original HuggingFace / LeRobot class, compiles that
submodule, then restores the method (dryrun leaves the replacement in place).

``@register_patch`` does not install anything. It records a factory and a dotted
class path on a backend (``"pi05"``, ``"groot"``, ``"nemotron"``).
``apply_patches(backend)`` imports that class and does
``setattr(Cls, "forward", factory(original))`` for the duration of
``export()``.

HuggingFace ``DynamoExporter`` uses the same two steps. The purpose is
different. HF patches make the original modeling ``forward`` traceable. Edge
patches change ``forward`` first so TensorRT traces plugin ops
(``torch.ops.trt.*``), not HuggingFace attention. After compile, the outer
graph is ``spec.run()`` → ``execute_engine``. There is no HF attention left
to patch, so the HuggingFace ``"dynamo"`` registry does not apply.

.. code-block:: python

    from exporters.plugin.attn_patches import register_patch

    PI05 = "pi05"

    @register_patch(
        PI05,
        "transformers.models.paligemma.modeling_paligemma.PaliGemmaModel.forward",
    )
    def _patch_paligemma_image_features(_original):
        def forward(self, pixel_values, **kwargs):
            image_outputs = self.vision_tower(pixel_values, **kwargs)
            hidden = image_outputs.last_hidden_state
            return self.multi_modal_projector(hidden)

        return forward

You compile the original submodule (``PaliGemmaModel``, ``PiGemmaModel``,
``FlowmatchingActionHead``, …), not a wrapper ``nn.Module``. The patched
``forward`` is what TensorRT traces: a tensor in, a tensor out, plugin
attention inside.

When the same class is used twice (PI05 language vs action expert), the
patched ``forward`` checks for Edge arguments such as
``rope_rotary_cos_sin``. If they are missing, the original HuggingFace
``forward`` runs:

.. code-block:: python

    def forward(self, inputs_embeds=None, rope_rotary_cos_sin=None, **kwargs):
        if rope_rotary_cos_sin is None:
            return original(self, inputs_embeds=inputs_embeds, **kwargs)
        return causal_lm_plugin_forward(self, inputs_embeds, rope_rotary_cos_sin, ...)

Attention patches follow the same rule: ``GemmaAttention.forward`` uses the
language plugin when ``rope_rotary_cos_sin`` is present, otherwise eager HF
attention.

The spec installs the family once around the component loop:

.. code-block:: python

    class Pi05Spec(EdgeSpec):
        def apply_patches(self, model=None):
            from exporters.plugin.attn_patches import apply_patches
            return apply_patches("pi05")

When the decorator is enough
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``type(module)`` is the class named in the dotted path,
``@register_patch`` plus ``apply_patches`` is all you need. That is Siglip,
Qwen3, Llama, ``GR00TN15``, the action head, and so on.

When it is not
^^^^^^^^^^^^^^

The path must be the **same class object** as the live module. A look-alike
file under another import is a different class. ``setattr`` on one does not
change the other.

``trust_remote_code=True`` downloads Hub ``.py`` files into HuggingFace's
module cache and imports them. ``AutoModel.from_config`` then builds an
instance in memory. That class's module path looks like
``transformers_modules.<repo>.<hash>.modeling_...``. It is not stable and
does not exist until load, so the decorator cannot name it. The cache stores
source, not the ``nn.Module``.

GR00T's Eagle is this case. The LeRobot path
``lerobot.policies.groot.eagle2_hg_model....Eagle25VLForConditionalGeneration``
is a different class from the HuggingFace cache copy that ``from_config``
actually constructs.

Live-object patches
^^^^^^^^^^^^^^^^^^^

``apply_groot_patches(model)`` is not a second decorator. It is the place
that has the instance, so it can patch ``type(eagle_model)``. It still runs
``apply_patches("groot")`` for every class that has a stable path.

Nemotron's ``apply_nemotron_patches(model)`` is the same idea for mixers:
the registry is string paths; anything that only exists on the live object
needs ``model``. The compiled module is still the original
``NemotronHForCausalLM``, not a wrapper.

Add a new model
---------------

``EdgeExporter.export`` never branches on PI05 vs GR00T. It loads an ``EdgeSpec``
and loops ``spec.components``. A new architecture is a new spec plus a patch
backend.

Create ``tools/hf/exporters/models/<family>/``:

.. code-block:: text

    <family>/
        spec.py       # EdgeSpec: components, sample inputs, prepare, run
        patches.py    # @register_patch factories on this family's backend
        helpers.py    # optional packing / submodule lookup

Import the spec from ``models/__init__.py`` so registration happens when the
exporter package loads:

.. code-block:: python

    from hf.exporters.models.my_vla import spec as _my_vla  # noqa: F401

1. Register the spec
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from hf.exporters.spec import EdgeSpec, register_edge_spec

    @register_edge_spec("my_vla")
    class MyVlaSpec(EdgeSpec):
        components = ("vision", "language", "action")

        def apply_patches(self, model=None):
            from hf.exporters.plugin.attn_patches import apply_patches
            from .patches import MY_VLA
            return apply_patches(MY_VLA)

Add a heuristic in ``infer_model_type`` (in ``exporters/spec.py``) if you want
``model_type`` to be optional, or always pass ``EdgeConfig(model_type="my_vla")``.

2. Patch original ``forward`` methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``patches.py``, register one factory per class you need to change. Put the
Edge I/O in that ``forward``. Compile the original module; do not introduce a
wrapper class.

Reuse the shared plugin attention factories when the layout matches
(``_patch_vision_attention``, ``_patch_language_attention``). Reuse
``causal_lm_plugin_forward`` for decoder-only prefill.

3. Select submodules and flatten I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``prepare(name, model, sample, upstream, config)`` returns a ``ComponentBundle``:

* ``module`` — the original submodule to compile (vision tower, decoder, action head)
* ``trace_args`` / ``save_args`` — positional tensors for ``torch.export`` / the engine
* ``input_names`` / ``output_names`` — written into ``config.json``
* ``context_attention_mask_type`` — padding vs causal for the language plugin

``capture_upstream`` maps this engine's outputs into keys the next ``prepare``
needs (image tokens, prefix KV, context embeddings).

4. Call engines in ``run()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``run(engines, sample)`` is the outer graph. Call ``call_engine`` for each
component and keep packing in Python (``fuse_prefix``, ``scatter_image_tokens``,
or your own custom op).

.. code-block:: python

    def run(self, engines, sample):
        vis = call_engine(engines["vision"], "vision", sample["pixel_values"])[0]
        prefix = fuse_prefix(vis, sample["lang_embeds"], sample["compact_index"])
        lm = call_engine(engines["language"], "language", prefix, ...)
        return call_engine(
            engines["action"], "action", sample["step_actions"], ..., lm[2], lm[3]
        )

5. Collate sample inputs
^^^^^^^^^^^^^^^^^^^^^^^^

``prepare_sample_inputs`` turns the caller payload into the stem dict ``prepare``
and ``run`` share. If the caller already passed tensors (``pixel_values``,
``input_ids``, …), return them. Otherwise load a preprocessor / tokenizer here
so the example scripts can pass only ``device`` and ``dtype``.

Example scripts
---------------

The smoke scripts live next to the other Dynamo examples. Default is **dryrun**
(no TensorRT). Pass ``--compile`` to build engines.

.. code-block:: bash

    cd TensorRT/examples/dynamo

    python run_pi05_export.py
    python run_pi05_export.py --compile --engine-dir /tmp/pi05_edge

    python run_groot_export.py
    python run_groot_export.py --compile --engine-dir /tmp/groot_edge

    python run_nemotron_export.py --prompt "Hello."
    python run_nemotron_export.py --compile --checkpoint nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Script
     - What to pass
     - Engines
   * - ``run_pi05_export.py``
     - LeRobot ``PI05Policy``
     - vision, language, action
   * - ``run_groot_export.py``
     - LeRobot ``GrootPolicy``
     - vision, language, context_projection, action
   * - ``run_nemotron_export.py``
     - HuggingFace ``NemotronHForCausalLM``
     - language

Each script loads plugins, forces ``eager`` attention, calls ``EdgeExporter.export``,
prints ``exporter.engines``, and runs the returned program once.

To export only one component while you debug a family, set
``EdgeConfig(components=("vision",))`` (or pass a subset of ``spec.components``).

Configuration
-------------

``EdgeConfig`` knobs that matter for this path:

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Field
     - Default
     - Role
   * - ``engine_dir``
     - ``"edge_engines"``
     - Output directory; one subdirectory per component
   * - ``dryrun``
     - ``False``
     - Skip TensorRT; keep patched Python modules for ``execute_engine``
   * - ``skip_runtime_export``
     - ``False``
     - Return the runtime module without ``torch.export`` of the outer graph
   * - ``model_type``
     - inferred
     - ``"pi05"``, ``"groot"``, ``"nemotron_h"``
   * - ``components``
     - spec default
     - Subset of engines to compile
   * - ``max_seq_len``
     - ``968``
     - KV / RoPE capacity for language
   * - ``trt_settings``
     - ``{}``
     - Forwarded into ``torch_tensorrt.dynamo.compile``

``strict``, ``dynamic``, and ``dynamic_shapes`` match HuggingFace ``DynamoConfig``
and apply to the **outer** ``torch.export`` of the runtime, not to the
per-component TensorRT compiles.

Plugins vs ``execute_engine``
-----------------------------

There are two different custom-op namespaces. Do not mix them up.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Where
     - Ops
     - What you see
   * - Outer ``ExportedProgram``
     - ``edge_llm::execute_engine``, ``fuse_prefix``, ``scatter_image_tokens``
     - ``print(program.graph)`` after ``EdgeExporter.export``
   * - Inside one component engine
     - ``trt::attention_plugin``, ``trt::vit_attention_plugin``,
       ``trt::causal_conv1d``, ``trt::update_ssm_state``,
       ``trt::nvfp4_moe_plugin``
     - The FX graph of that component during ``dynamo.compile``

A patched attention ``forward`` calls ``torch.ops.trt.attention_plugin``.
When that component is ``torch.export`` + ``torch_tensorrt.dynamo.compile``'d,
the plugin converter turns that op into an ``AttentionPlugin`` layer inside
that engine. The outer ``ExportedProgram`` never sees it; it only sees
``execute_engine("language")``.


Plugin Converters
-----------------

A **plugin converter** is the Torch-TensorRT dynamo hook that maps a
``torch.ops.trt.*`` node onto a TensorRT ``IPluginV3`` (from
``libNvInfer_edgellm_plugin.so``).

Without a converter, ``dynamo.compile`` cannot lower the custom op and
either graph-breaks or fails. With a converter, the op becomes one TensorRT
plugin layer.

Converters live in
``tools/hf/exporters/plugin/plugin_converter.py`` and are
registered with ``@dynamo_tensorrt_converter``. Example for ViT attention:

.. code-block:: python

   @dynamo_tensorrt_converter(
       torch.ops.trt.vit_attention_plugin.default,
       supports_dynamic_shapes=True,
       priority=ConverterPriority.HIGH,
   )
   def convert_vit_attention_plugin(ctx, target, args, kwargs, name):
       creator = get_trt_plugin_creator("ViTAttentionPlugin", "1", "")
       plugin = creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
       layer = ctx.net.add_plugin_v3(inputs, [], plugin)
       return layer.get_output(0)

That is: look up the plugin by TensorRT name, fill ``PluginField``s, add the
layer, return its outputs. ``load_plugins_for_trt()`` imports this module so
the converters are registered before compile.


How to add a plugin
-------------------

You need four pieces. The C++ plugin in Edge-LLM is assumed to already
exist and be in ``libNvInfer_edgellm_plugin.so``.

**1. Torch custom op (eager + fake)** so Dynamo can trace.

Register in ``plugin_utils.py``, ``mamba.py``, or ``moe.py``:

.. code-block:: python

   @torch.library.custom_op("trt::my_plugin", mutates_args=())
   def my_plugin(x: torch.Tensor, ...) -> torch.Tensor:
       return _my_plugin_eager(x, ...)

   @my_plugin.register_fake
   def _(x, ...):
       return torch.empty_like(x)

Call ``load_plugins_for_trt()`` so this op exists before export.

**2. Call it from a patched ``forward``.**

In ``models/<family>/patches.py``, ``@register_patch`` a class ``forward``
that emits ``torch.ops.trt.my_plugin`` when Edge I/O is present. That is
how attention already works: the plugin call is in the patched
``GemmaAttention.forward``, not in a wrapper module.

**3. Plugin converter.**

Add ``@dynamo_tensorrt_converter(torch.ops.trt.my_plugin.default)`` in
``plugin_converter.py``. It must use the same TensorRT plugin name / version
the ``.so`` registered (see ``get_trt_plugin_creator``).

**4. Load the ``.so``.**

``load_plugins_for_trt()`` already calls ``load_plugin()``, which
``ctypes.CDLL``s ``EDGE_LLM_PLUGIN_SO``. After that, TensorRT can create
the plugin during ``dynamo.compile``.

Checklist for a new kernel:

* Eager op + fake kernel (traceable).
* Patched ``forward`` that actually calls the op.
* Converter that adds the TRT plugin layer.
* Plugin present in ``libNvInfer_edgellm_plugin.so``.
* ``load_plugins_for_trt()`` before ``EdgeExporter().export``.
