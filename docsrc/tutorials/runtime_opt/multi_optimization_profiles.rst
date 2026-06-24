.. _multi_optimization_profiles_tutorial:

Multiple Optimization Profiles (Prefill / Decode)
=================================================

TensorRT tunes kernels for the **optimization profile** of an engine: a
``[min, opt, max]`` range for every dynamic input dimension. Kernels are tuned at
the ``opt`` point, so a single profile can only be optimal for one shape.

Many models, however, run in several distinct shape *regimes* that share the same
weights. The canonical case is an autoregressive LLM:

* **prefill** -- the prompt is processed in one shot, so ``seq`` is large, and
* **decode** -- tokens are generated one at a time, so ``seq == 1``.

With a single dynamic range ``seq in [1, max]`` you must pick one ``opt``. Tuning
for the long prefill length leaves **decode** -- the latency-critical, most
frequently executed phase -- running on kernels chosen for a sequence length it
never sees.

Torch-TensorRT lets you declare **multiple optimization profiles** on a single
input and select the active one at runtime. The engine is built once and each
profile is tuned independently.

Declaring profiles
------------------

Pass an ordered list of ``{"min", "opt", "max"}`` dicts to
:class:`torch_tensorrt.Input` via ``profiles``. The **list index** is the
optimization-profile index you select at runtime.

.. code-block:: python

    import torch
    import torch_tensorrt

    DECODE_IDX, PREFILL_IDX = 0, 1

    profiled_input = torch_tensorrt.Input(
        dtype=torch.int64,
        profiles=[
            # index 0 -> decode: seq pinned to 1 (a fully static profile)
            {"min": (1, 1), "opt": (1, 1), "max": (1, 1)},
            # index 1 -> prefill: seq in [1, 512], tuned at 256
            {"min": (1, 1), "opt": (1, 256), "max": (1, 512)},
        ],
    )

``profiles`` is mutually exclusive with the single-shape ``min_shape`` /
``opt_shape`` / ``max_shape`` (and ``shape``) arguments.

The **union envelope**
~~~~~~~~~~~~~~~~~~~~~~~~

``torch.export`` traces a model over one ``[min, opt, max]`` range, so
``Input`` automatically derives the **union envelope** of all profiles
(elementwise ``min`` of every ``min`` and ``max`` of every ``max``; ``opt`` is
taken from the first profile). Each declared profile is a subset of this
envelope. You export over the envelope and the individual profiles become the
per-profile TensorRT tunings:

.. code-block:: python

    print(profiled_input.shape["min_shape"])  # (1, 1)
    print(profiled_input.shape["max_shape"])  # (1, 512)

Compile
-------

Export once over the union range, then compile as usual. Every input that
declares ``profiles`` must declare the **same number** of profiles; static
inputs (or dynamic inputs without ``profiles``) reuse their single shape in every
profile.

.. code-block:: python

    seq = torch.export.Dim("seq", min=1, max=512)
    exported = torch.export.export(model, (example_ids,), dynamic_shapes=({1: seq},))

    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=[profiled_input],
        enabled_precisions={torch.float16},
        min_block_size=1,
    )

Selecting a profile at runtime
------------------------------

Selection is **manual by default**. Use the
:func:`torch_tensorrt.runtime.optimization_profile` context manager to pin a
profile by index for the duration of a ``with`` block; the prior state is saved
on enter and restored on exit, so blocks nest cleanly.

.. code-block:: python

    from torch_tensorrt.runtime import optimization_profile

    with optimization_profile(trt_model, DECODE_IDX):
        logits = trt_model(decode_ids)      # seq == 1

    with optimization_profile(trt_model, PREFILL_IDX):
        logits = trt_model(prefill_ids)     # seq == 256

Pass ``"auto"`` to let Torch-TensorRT choose from the input shapes. Auto-selection
is **lazy / first-working**: it scans profiles in index order and uses the first
whose ``[min, max]`` contains the input. Order matters when profiles overlap --
declaring ``decode`` first lets it win the ``seq == 1`` overlap:

.. code-block:: python

    with optimization_profile(trt_model, "auto"):
        trt_model(decode_ids)    # seq == 1   -> index 0 (decode) accepts -> decode
        trt_model(prefill_ids)   # seq == 256 -> index 0 rejects -> index 1 (prefill)

Profiles, graph breaks, and serialization
-----------------------------------------

* **Graph breaks**: when a model is partitioned into several TensorRT engines,
  every engine carries the same number of profiles. Torch-TensorRT propagates the
  per-profile bounds across the break, evaluating any *derived* dynamic dimension
  (e.g. a ``reshape`` that turns ``seq`` into ``16 * seq``) through to the
  downstream engine, so runtime selection stays consistent for the whole module.
* **Serialization / runtimes**: profile state is reconstructed from the TensorRT
  API on load (``getNbOptimizationProfiles`` / ``getProfileShape``), so a
  serialized engine keeps its profiles with no extra metadata. The same
  ``optimization_profile`` API drives both the C++ and Python runtimes, which
  remain interchangeable.

Why it helps: a worked latency example
--------------------------------------

The example :ref:`multi_optimization_profiles` compiles ``google/gemma-3-1b-it``
twice -- once with a single profile (tuned at the prefill length) and once with
separate decode/prefill profiles -- then compares per-call latency. The
multi-profile engine dedicates a **static** profile (``seq`` pinned to 1) to
decode, letting TensorRT specialize that path (measured on an NVIDIA A40, FP16):

.. code-block:: text

    Per-call latency (ms), batch=1
    regime                single-profile   multi-profile   speedup
    --------------------------------------------------------------
    decode (seq=1)                 5.232           4.597     1.14x
    prefill (seq=128)              7.152           7.534     0.95x

Prefill is essentially unchanged (both engines tune it at the same ``opt``),
while decode -- the regime executed once per generated token -- is faster. Exact
numbers depend on the model and GPU; the takeaway is that one engine can be tuned
well for *both* regimes instead of compromising on a single ``opt`` shape.

.. note::

   Because the model has two dynamic inputs (``input_ids`` and ``position_ids``),
   the example passes one profiled ``Input`` for each, both declaring the same
   profiles. The HuggingFace attention path also needs a TensorRT-friendly SDPA
   lowering (``tools/llm/torchtrt_ext/register_sdpa``), and ``gemma-3-1b-it`` is a
   gated model requiring Hugging Face authentication.

.. seealso::

   - Runnable example: :ref:`multi_optimization_profiles`
   - :class:`torch_tensorrt.Input`
   - :func:`torch_tensorrt.runtime.optimization_profile`
