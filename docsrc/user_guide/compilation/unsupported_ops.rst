.. _unsupported_ops:

Handling Unsupported Operators
================================

When Torch-TensorRT encounters an operator it cannot convert to a TRT layer, it has
several options. This guide explains what happens by default and how to handle each case.

----

What Happens by Default
------------------------

Torch-TensorRT partitions the FX graph into TRT-convertible and non-convertible
subgraphs. Ops with no converter fall back to **PyTorch** automatically — they run in
eager mode as part of a ``PyTorch`` subgraph between two TRT subgraphs.

Use :ref:`dryrun` to see the partition layout before committing to a full compile:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        dryrun=True,
    )

The dryrun output shows which ops triggered fallback and where the TRT/PyTorch boundary
is. A few PyTorch fallback ops embedded in a large TRT block typically have negligible
performance impact.

----

Option 1: Accept the Fallback (Most Common)
-------------------------------------------

If an op runs correctly in PyTorch and the fallback is infrequent relative to the rest
of the model, just leave it. Many custom attention mechanisms, preprocessing ops, and
postprocessing steps fall into this category.

**FlashAttention** is a common example. The ``flash_attn`` package registers custom
CUDA kernels that are not in TRT's op set. Torch-TensorRT will fall back those kernels
to PyTorch, while the rest of the model (embeddings, FFN layers, etc.) runs in TRT:

.. code-block:: python

    import torch
    import torch_tensorrt
    from flash_attn import flash_attn_func  # custom CUDA kernel

    class ModelWithFlashAttn(torch.nn.Module):
        def forward(self, q, k, v, x):
            # flash_attn falls back to PyTorch automatically
            attn_out = flash_attn_func(q, k, v)
            # other ops can still run in TRT
            return self.ffn(attn_out + x)

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        min_block_size=1,
    )

If you want to explicitly list ops that should always fall back to PyTorch (to prevent
them from being incorrectly converted), use ``torch_executed_ops``:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        torch_executed_ops={
            "torch.ops.flash_attn.flash_attn_func",
        },
    )

----

Option 2: Write a Custom Converter
------------------------------------

If the op falls back but you want TRT to run it (for performance or to avoid a graph
break), you can register a custom converter that maps the op to TRT layers.

**Automatic converter generation** — Torch-TensorRT can generate a converter template
from a PyTorch op registration:

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt.dynamo.conversion.converter_utils import auto_generate_converter

    # Automatically creates a converter using TRT's standard layer set
    @torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(
        torch.ops.my_package.my_op.default
    )
    def my_op_converter(ctx, target, args, kwargs, name):
        input_tensor = args[0]
        # map to TRT layers
        layer = ctx.net.add_activation(input_tensor, trt.ActivationType.RELU)
        return layer.get_output(0)

See the :ref:`auto_generate_converters` example for a complete walkthrough.

----

Option 3: Use a Custom TensorRT Plugin
----------------------------------------

For ops that cannot be expressed as a composition of TRT layers (custom CUDA kernels,
Triton kernels, architecture-specific ops), register a TRT plugin.

Torch-TensorRT provides two plugin mechanisms:

**Python-based plugins** (simpler, Python runtime only):

.. code-block:: python

    from torch_tensorrt.dynamo.conversion import TRTInterpreter
    import tensorrt as trt

    # Register a custom TRT plugin for a Triton kernel
    # See the custom_kernel_plugins example for the full pattern

**AOT (ahead-of-time) compiled plugins** (supports C++ runtime and serialized engines):

AOT plugins compile the kernel to a shared library at build time. This is the only way
to include a custom kernel in a serialized engine that runs in C++ without Python.

See the :ref:`aot_plugin` and :ref:`nvrtc_aot_plugin` examples for complete code.

----

Option 4: Restructure the Model
---------------------------------

Sometimes an unsupported op can be replaced with an equivalent supported op.

**Scaled Dot-Product Attention (SDPA)**

``torch.nn.functional.scaled_dot_product_attention`` has a TRT converter. However,
PyTorch often dispatches SDPA to backend-specific implementations such as
``_scaled_dot_product_flash_attention`` or ``_scaled_dot_product_efficient_attention``
that appear as separate ops in the exported graph and have no TRT converter.

There are two scenarios:

*Scenario A — your own model code calls a third-party flash_attn package:*

Replace the direct ``flash_attn_func`` call with ``F.scaled_dot_product_attention``:

.. code-block:: python

    import torch.nn.functional as F

    # Before (unsupported in TRT):
    # from flash_attn import flash_attn_func
    # attn_out = flash_attn_func(q, k, v)

    # After (TRT-compatible):
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

*Scenario B — a HuggingFace model internally selects a backend-specific SDPA kernel:*

Load the model with ``attn_implementation="sdpa"`` and register an FX lowering pass
that rewrites the backend-specific ops back to the generic
``F.scaled_dot_product_attention`` before TRT compilation. This is the pattern used
in ``tools/llm/run_llm.py``:

.. code-block:: python

    from transformers import AutoModelForCausalLM
    from torch_tensorrt.dynamo.lowering import TORCH_TRT_DECOMPOSITIONS
    from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import _aten_lowering_pass
    from torch_tensorrt.dynamo._settings import CompilationSettings

    # 1. Load the model using PyTorch's built-in SDPA (not flash_attn)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        attn_implementation="sdpa",
    ).eval().cuda()

    # 2. Remove Torch-TensorRT's SDPA decompositions so the op reaches the converter
    _SDPA_VARIANTS = [
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    ]
    for op in _SDPA_VARIANTS:
        TORCH_TRT_DECOMPOSITIONS.pop(op, None)

    # 3. Register a lowering pass to rewrite backend-specific SDPA to the generic form
    @_aten_lowering_pass(index=0)
    def sdpa_rewrite_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        for node in list(gm.graph.nodes):
            if node.op == "call_function" and node.target in _SDPA_VARIANTS:
                q, k, v = node.args[0], node.args[1], node.args[2]
                with gm.graph.inserting_after(node):
                    new_node = gm.graph.call_function(
                        torch.nn.functional.scaled_dot_product_attention,
                        args=(q, k, v),
                        kwargs={"is_causal": True},
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()
        return gm

    # 4. Compile as normal
    trt_model = torch_tensorrt.compile(model, arg_inputs=inputs)

For a complete, production-ready implementation of this pattern across LLaMA, Qwen,
and Gemma3 models (including sliding-window attention), see
``tools/llm/torchtrt_ext/register_sdpa.py``.

**Other common restructuring patterns**

* Custom ``LayerNorm`` implementations — TRT has a native group norm / layer norm op.
  Replacing a hand-written ``(x - mean) / (std + eps)`` with ``torch.nn.LayerNorm``
  typically produces a single fused TRT layer.

* ``torch.einsum`` — TRT supports a limited subset of einsum patterns. If your einsum
  pattern is unsupported, expand it into ``matmul`` + ``transpose`` + ``sum`` ops which
  all have converters.

----

Checking Coverage Before Compilation
--------------------------------------

Use ``dryrun=True`` to see a report of what will fall back without triggering a full TRT
engine build:

.. code-block:: python

    torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs, dryrun=True)

Use :ref:`supported_ops` for a full list of ATen ops with TRT converters.

To query programmatically whether a specific op has a converter:

.. code-block:: python

    from torch_tensorrt.dynamo.conversion import DYNAMO_CONVERTERS

    op = torch.ops.aten.gelu.default
    print(op in DYNAMO_CONVERTERS)  # True/False

----

FAQ
---

**"My model has a custom C++/CUDA extension. Will it work?"**

    Custom C++ extensions registered via ``torch.library`` or ``TORCH_LIBRARY`` are
    opaque to TRT — they fall back to PyTorch automatically. If the extension is
    performance-critical, write an AOT plugin (see :ref:`aot_plugin`).

**"FlashAttention makes my model fail export entirely"**

    ``torch.export.export`` in strict mode traces through Python-level control flow but
    needs to trace through all ops. Try:

    * ``strict=False`` in ``torch.export.export`` (non-strict tracing).
    * ``torch_tensorrt.dynamo.trace(model, inputs, strict=False)``.

    If the export still fails, the FlashAttention op may register custom autograd
    functions that can't be traced. Use the ``torch.nn.functional.scaled_dot_product_attention``
    (SDPA) path instead — it is supported by both ``torch.export`` and Torch-TensorRT.

**"I have a Triton kernel. Can I use it in a serialized TRT engine?"**

    Triton kernels can be wrapped as TRT plugins via the AOT plugin path. See
    :ref:`aot_plugin` for an end-to-end example of compiling a Triton kernel as a
    TRT plugin for use in a serialized engine.

**"Operator X is listed as supported but my model still falls back"**

    The converter may only support specific overloads or dtype combinations. Check:

    * The exact overload name in the dryrun output (e.g. ``aten::add.Tensor`` vs
      ``aten::add.Scalar``).
    * The input dtype — some converters only handle ``float16`` or ``float32``.
    * ``min_block_size`` — if the only unsupported op is surrounded by very few
      supported ops, TRT may choose to keep the whole small subgraph in PyTorch.

    File a bug at https://github.com/pytorch/TensorRT/issues with the dryrun output
    and model snippet.
