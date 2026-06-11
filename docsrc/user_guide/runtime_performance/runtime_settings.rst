.. _runtime_settings:

Runtime Settings (TensorRT-RTX)
================================

Three knobs that affect TensorRT-RTX runtime behavior **without** recompiling:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Type
     - Effect
   * - ``cuda_graph_strategy``
     - ``"disabled"`` | ``"whole_graph_capture"``
     - Whether TensorRT-RTX captures+replays the engine internally.
   * - ``dynamic_shapes_kernel_specialization_strategy``
     - ``"lazy"`` | ``"eager"`` | ``"none"``
     - When dynamic-shape kernels are JIT-compiled.
   * - ``runtime_cache``
     - ``None`` | ``str`` path | ``RuntimeCacheHandle``
     - On-disk cache of JIT-compiled kernels.

All three live on ``torch_tensorrt.runtime.RuntimeSettings`` (a frozen
dataclass). All three are **TensorRT-RTX only** — they are no-ops on standard
TensorRT, and constructing a non-default ``RuntimeSettings`` on a non-RTX build
emits a ``UserWarning``.

----

The three ways to apply settings
--------------------------------

Direct assignment — permanent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt.runtime import RuntimeSettings

    mod = torch_tensorrt.compile(model, inputs=inputs)
    mod.runtime_settings = RuntimeSettings(runtime_cache="/var/cache/jit.bin")

Use when you want the setting to apply for the module's lifetime.

``runtime_config(...)`` context manager — scoped override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_tensorrt.runtime import runtime_config

    with runtime_config(mod, cuda_graph_strategy="whole_graph_capture"):
        out = mod(x)
    # settings restored on exit

Use when you want to flip a setting just for one call site. The CM snapshots
prior settings on enter and restores them on exit.

``runtime_config`` accepts a single module or a list:

.. code-block:: python

    with runtime_config([mod_a, mod_b], cuda_graph_strategy="whole_graph_capture") as (a, b):
        out_a = a(x)
        out_b = b(x)

A sugar wrapper exists for the dynamic-shapes strategy field:

.. code-block:: python

    from torch_tensorrt.runtime import set_dynamic_shapes_kernel_strategy

    with set_dynamic_shapes_kernel_strategy(mod, "eager"):
        out = mod(x)

For the ``cuda_graph_strategy`` field, prefer
``enable_cudagraphs(mod, cuda_graph_strategy=...)`` (see
:ref:`runtime_settings_combining_cudagraphs` below). There is no
``set_cuda_graph_strategy(...)`` wrapper — flipping ``cuda_graph_strategy`` is
almost always paired with ``enable_cudagraphs``, so the two CMs are collapsed
into one.

``runtime_cache(...)`` context manager — shared cache + IO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torch_tensorrt.runtime import runtime_cache

    # Cache implicitly attached to mod for the duration of the with-block.
    # Loaded from disk on enter, saved on exit.
    with runtime_cache(mod, "/var/cache/jit.bin"):
        out = mod(x)

Bind the handle (``as rc``) when you need to do something with it — pass it
to nested calls, inspect it, or save it mid-block:

.. code-block:: python

    with runtime_cache(mod, "/var/cache/jit.bin") as rc:
        out = mod(x)
        # mid-block checkpoint
        rc.save("/var/cache/jit-mid.bin")

Different from ``runtime_config(runtime_cache="...")``:

* ``runtime_config`` gives each module its **own** implicit cache; modules do
  not share kernels.
* ``runtime_cache`` constructs **one shared** ``RuntimeCacheHandle`` and
  attaches it to *all* listed modules.

Use ``runtime_cache`` when you want multiple modules to pool kernels, or you
want explicit I/O control (load before, save after).

Stream-backed ``runtime_cache``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``path`` can be a file-like object (``io.BytesIO``, opened binary file). This
is useful for in-memory round-trips, gzipped storage, or network sinks:

.. code-block:: python

    import io

    buf = io.BytesIO()
    with runtime_cache(mod, buf):
        out = mod(x)
    # buf.getvalue() holds the saved bytes; the caller owns open/close.

Stream-mode behavior:

* On enter: ``stream.read()`` once, bytes deserialized into the cache.
* On exit: cache serialized, ``stream.write(bytes)`` once.
* ``rc.path`` reports ``""`` in stream-mode.

----

Composing the context managers
------------------------------

Idiomatic: cache outside, strategy inside
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you bind the handle as ``rc``, plug it into the nested ``runtime_config``
call so it is clear which cache the inner override applies to:

.. code-block:: python

    with runtime_cache(mod, "/var/cache/jit.bin") as rc:
        with runtime_config(mod, runtime_cache=rc, cuda_graph_strategy="whole_graph_capture"):
            out = mod(x)

If you are not passing ``rc`` anywhere, drop the binding — the cache is
attached implicitly by the outer CM regardless:

.. code-block:: python

    with runtime_cache(mod, "/var/cache/jit.bin"):
        with runtime_config(mod, cuda_graph_strategy="whole_graph_capture"):
            out = mod(x)

Both forms produce identical engine state. The explicit form is preferable
when readability matters (multi-module composition, deep nesting); the
implicit form is fine for one-off scripts. The cache lives "longer" than
transient strategy toggles in either case — the strategy CM's snapshot
captures the cache-attached state, applies the override, restores the
snapshot on exit.

.. _runtime_settings_combining_cudagraphs:

Combining with ``enable_cudagraphs(...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enable_cudagraphs(mod, cuda_graph_strategy="whole_graph_capture")`` applies
the RTX cuda-graph strategy and wraps the module in one CM — exactly one
``createExecutionContext`` call:

.. code-block:: python

    from torch_tensorrt.runtime import enable_cudagraphs

    with enable_cudagraphs(mod, cuda_graph_strategy="whole_graph_capture") as wrapped:
        out = wrapped(x)

Under the hood, ``enable_cudagraphs`` opens a
``runtime_config(mod, cuda_graph_strategy=...)`` CM *before* the wrapper's
``warm_up()`` materializes the engine's ``IExecutionContext``, then closes it
after teardown on exit. The strategy is in effect for the captured context
but restored on the engines once the wrapper is gone.

The ``cuda_graph_strategy`` kwarg is **TensorRT-RTX only**; passing it on a
non-RTX build raises ``RuntimeError`` at the call site.

If you want *non*-strategy knobs alongside cudagraphs (e.g. a different
``dynamic_shapes_kernel_specialization_strategy``), nest ``runtime_config``
*outside* ``enable_cudagraphs``:

.. code-block:: python

    from torch_tensorrt.runtime import runtime_config, enable_cudagraphs

    with runtime_config(mod, dynamic_shapes_kernel_specialization_strategy="eager"):
        with enable_cudagraphs(mod, cuda_graph_strategy="whole_graph_capture") as wrapped:
            out = wrapped(x)

.. warning::

   Any setting flipped *inside* ``enable_cudagraphs(...)`` invalidates the
   warmed ``IExecutionContext`` and forces a re-JIT on RTX. Apply your
   settings outside (via ``runtime_config(...)`` or the
   ``cuda_graph_strategy=`` kwarg on ``enable_cudagraphs`` itself), not
   inside.

Share one cache across ``runtime_config(...)`` calls on different modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The handle yielded by ``runtime_cache([...])`` is a ``RuntimeCacheHandle`` you
can plug into any ``runtime_config(...)`` call. Useful when you want a shared
cache plus per-module strategy overrides:

.. code-block:: python

    with runtime_cache([mod_a, mod_b], "/var/cache/jit.bin") as rc:
        # mod_a gets whole-graph capture; mod_b stays default.
        # Both modules continue sharing rc.
        with runtime_config(mod_a, runtime_cache=rc, cuda_graph_strategy="whole_graph_capture"):
            out_a = mod_a(x)
            out_b = mod_b(x)

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

A two-stage pipeline where ``mod1`` and ``mod2`` share a JIT-kernel cache,
``mod1`` runs with a temporary dynamic-shapes override + cudagraph capture,
and ``mod2`` consumes ``mod1``'s output under the same cache:

.. code-block:: python

    from torch_tensorrt.runtime import (
        runtime_cache,
        runtime_config,
        enable_cudagraphs,
    )

    with runtime_cache([mod1, mod2], "/var/cache/jit.bin") as rc:
        with (
            runtime_config(
                mod1,
                runtime_cache=rc,
                dynamic_shapes_kernel_specialization_strategy="eager",
            ) as modr,
            enable_cudagraphs(modr, cuda_graph_strategy="whole_graph_capture") as cg,
        ):
            outputs = cg(*inputs)
        mod2(*outputs)

What happens, step by step:

* The outer ``runtime_cache`` builds **one** shared ``RuntimeCacheHandle``
  ``rc`` and attaches it to both ``mod1`` and ``mod2`` — any kernel JIT'd
  while running ``mod1`` is available to ``mod2`` with no re-compile.
* The inner ``runtime_config`` applies ``"eager"`` dynamic-shapes
  specialization to ``mod1`` for this scope and explicitly threads ``rc``
  through (the ``modr`` binding is just ``mod1`` with the override active).
* ``enable_cudagraphs(modr, cuda_graph_strategy="whole_graph_capture")``
  applies the RTX cuda-graph strategy and wraps ``modr`` for capture in one
  CM — one ``createExecutionContext`` call total.
* ``mod2(*outputs)`` runs *outside* the strategy + cudagraph scope but
  *inside* the cache scope, so it sees default settings plus the shared
  cache.
* On exit: cudagraph wrapper torn down → ``mod1``'s strategy restored →
  cache saved to ``/var/cache/jit.bin``.

----

Advanced: caller-owned ``RuntimeCacheHandle``
---------------------------------------------

Construct your own handle if you want full lifetime control:

.. code-block:: python

    from torch_tensorrt.runtime import RuntimeCacheHandle, RuntimeSettings

    handle = RuntimeCacheHandle(path="/var/cache/jit.bin", autosave_on_del=True)
    mod.runtime_settings = RuntimeSettings(runtime_cache=handle)
    out = mod(x)
    # handle.save() will fire when handle goes out of scope (autosave_on_del=True)

Or with explicit save/load:

.. code-block:: python

    handle = RuntimeCacheHandle(path="/var/cache/jit.bin")  # autosave_on_del=False default
    handle.load()
    mod.runtime_settings = RuntimeSettings(runtime_cache=handle)
    out = mod(x)
    handle.save()

----

Best practices
--------------

Apply settings *before* first execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IExecutionContext`` is created lazily on first execute. Apply settings
before that and you get **one** context create:

.. code-block:: python

    mod = torch_tensorrt.compile(...)
    mod.runtime_settings = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
    out = mod(x)  # single createExecutionContext call here

Apply settings *after* first execute and you get **two**:

.. code-block:: python

    mod = torch_tensorrt.compile(...)
    out = mod(x)  # context created with defaults
    mod.runtime_settings = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
    out = mod(x)  # context invalidated + recreated

On RTX, each ``createExecutionContext`` JIT-compiles the specialized kernel
set, so this matters for setup latency.

NCCL engines pay the extra create
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NCCL-collective engines eagerly materialize the context at setup (cross-rank
barrier ordering). Any subsequent ``mod.runtime_settings = ...`` triggers a
second create. This is a documented trade-off — apply settings before any
inference if you can, but the eager bind is non-negotiable for NCCL safety.

Default ``runtime_cache`` is shared per-user — concurrent processes can lose kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RuntimeSettings()`` defaults ``runtime_cache`` to
``{tmpdir}/torch_tensorrt_{user}/runtime_cache.bin``. The file is protected
by a filelock against corruption, but **not** against lost-update races::

    proc A: load -> {a1,a2}; generate {a3}; save  -> file = {a1,a2,a3}
    proc B: load (before A's save) -> {a1,a2}; generate {b1}; save -> file = {a1,a2,b1}
                                                                      # a3 lost

For hyperparameter sweeps, or any multi-process workload where lost kernels
matter:

.. code-block:: python

    # Option 1: per-worker path
    mod.runtime_settings = RuntimeSettings(runtime_cache=f"/var/cache/jit-worker-{worker_id}.bin")

    # Option 2: opt out
    mod.runtime_settings = RuntimeSettings(runtime_cache=None)

Don't nest ``runtime_cache(...)`` CMs with the same path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # DON'T:
    with runtime_cache(mod, "/p") as rc1:
        with runtime_cache(mod, "/p") as rc2:
            out = mod(x)

Each ``runtime_cache(...)`` builds a *different* ``RuntimeCacheHandle``
object. The inner one displaces the outer's handle from the engine. On inner
exit, ``rc2.save()`` writes ``/p``. On outer exit, the engine has ``rc1``
re-attached (different ``IRuntimeCache`` from ``rc2``), and ``rc1.save()``
overwrites ``/p`` with the now-stale ``rc1`` state. **Last writer wins;
mid-block kernels are silently lost.**

Setter is per-``TorchTensorRTModule``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mod.runtime_settings = rs`` only affects ``self``. If you compile a model
with multiple TRT subgraphs, walk the submodules:

.. code-block:: python

    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    for _, sub in compiled.named_modules():
        if isinstance(sub, TorchTensorRTModule):
            sub.runtime_settings = RuntimeSettings(...)

``runtime_config(...)`` and ``runtime_cache(...)`` do this walk automatically
— that is the easier API for compound models.

Non-TensorRT-RTX builds emit a warning, do nothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructing a non-default ``RuntimeSettings()`` on a non-RTX build emits a
``UserWarning`` and the settings have no effect. The dispatch path still
runs; it is just a no-op on the engine side. If you are shipping
cross-RTX/non-RTX code, you can suppress the warning with
``warnings.simplefilter("once", UserWarning)``.

----

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Goal
     - API
   * - Set a runtime knob permanently on one module
     - ``mod.runtime_settings = RuntimeSettings(...)``
   * - Temporary override for one call site
     - ``with runtime_config(mod, **overrides):``
   * - Just the dynamic-shapes kernel strategy
     - ``with set_dynamic_shapes_kernel_strategy(mod, "..."):``
   * - RTX cuda-graph strategy + cudagraphs capture in one CM
     - ``with enable_cudagraphs(mod, cuda_graph_strategy="..."):``
   * - Share one cache across multiple modules
     - ``with runtime_cache([a, b], path) as rc:``
   * - Cache to/from a stream
     - ``with runtime_cache(mod, io.BytesIO()):``
   * - Caller-controlled cache lifetime
     - construct ``RuntimeCacheHandle(...)`` directly
   * - In-memory cache (no disk)
     - ``RuntimeSettings(runtime_cache=None)`` or ``runtime_cache(mod, "")``
   * - Non-cuda-graph settings alongside cudagraphs capture
     - nest ``runtime_config(...)`` *outside* ``enable_cudagraphs(...)``
