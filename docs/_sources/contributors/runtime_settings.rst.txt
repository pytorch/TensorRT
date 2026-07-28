.. _runtime_settings_internals:

Runtime Settings
================

This page documents the design of the runtime-settings subsystem: the three
runtime-only knobs it governs, why they live outside ``CompilationSettings``,
and how they are applied across Torch-TensorRT's two runtimes. It is the
maintainer-facing counterpart to the user guide at
:doc:`/user_guide/runtime_performance/runtime_settings`, which covers the
public API.

.. note::

   Runtime settings only take effect on TensorRT-RTX builds. On standard
   TensorRT the knobs validate and are stored, but have no runtime effect.

Goal
----

Three knobs control *how the runtime drives an already-built engine*, without
changing the engine itself:

* ``cuda_graph_strategy`` — whether TensorRT-RTX captures the engine into a CUDA graph.
* ``dynamic_shapes_kernel_specialization_strategy`` — how kernels specialize when inputs have dynamic dimensions.
* ``runtime_cache`` — the TensorRT-RTX runtime kernel cache, a JIT-kernel cache persisted across runs.

These are **runtime concerns**, not compile-time ones: they do not affect an
engine's identity or its serialized form, only its execution. They are
therefore kept out of ``CompilationSettings`` and are not baked into the
serialized engine — instead they are held in memory, can be changed at any
time, and are sampled when the engine's ``IExecutionContext`` is (re)created.

Architecture
------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │ Public API                                                   │
   │   runtime_config · runtime_cache · enable_cudagraphs(...)    │
   │   · mod.runtime_settings = RuntimeSettings(...)              │
   └───────────────────────────────┬──────────────────────────────┘
                                   ▼
              RuntimeSettings  (frozen dataclass)
                cuda_graph_strategy
                dynamic_shapes_kernel_specialization_strategy
                runtime_cache : None | str | RuntimeCache
                                    │
                                    ▼
              TorchTensorRTModule ───── owns ─────▶ RuntimeCache (facade)
              single dispatch point                 the only handle users touch;
                                    │                forwards to one inner handle
                          cpp runtime loaded?
                    ┌───────────────┴───────────────┐
                 No │                               │ Yes
                    ▼                               ▼
           Python runtime path              C++ runtime path
           TRTEngine                        torch.classes.tensorrt.Engine
                │                                 │
                ▼                                 ▼
           TRTRuntimeConfig (py)            TRTRuntimeConfig (C++)
           owns trt.IRuntimeConfig          owns nvinfer1::IRuntimeConfig
                │                                 │
                ▼                                 ▼
           _RuntimeCacheHandle              RuntimeCacheHandle (torchbind)
           pure-Python inner                C++ inner (used directly)
                ▲                                               ▲
                └───  RuntimeCache._handle → matching inner  ───┘

Settings flow top-down. The two runtime columns are mirror images : engine ->
config shim -> inner cache handle, and the ``RuntimeCache`` facade (the only
handle users touch) forwards uniformly to whichever inner handle matches the
active runtime. The pieces are explained below.

User API
--------

The knobs are grouped in a frozen ``RuntimeSettings`` dataclass and applied
through context managers (scoped) or direct assignment (persistent):

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt.runtime import RuntimeSettings, runtime_config, runtime_cache

    # Scoped override for a with-block:
    with runtime_config(trt_model, cuda_graph_strategy="whole_graph_capture"):
        out = trt_model(*inputs)

    # A shared, disk-backed kernel cache across every engine under the target:
    with runtime_cache(trt_model, "/path/to/cache.bin"):
        out = trt_model(*inputs)

    # Persistent assignment:
    trt_model.runtime_settings = RuntimeSettings(runtime_cache="/path/to/cache.bin")

See the user guide for the full surface, including
``enable_cudagraphs(..., cuda_graph_strategy=...)``.

Design
------

Runtime, not compile-time
^^^^^^^^^^^^^^^^^^^^^^^^^^

The defining decision is that these knobs are runtime state. An engine's
serialized bytes and its identity are independent of them, so the same engine
can be driven with a different strategy or cache without recompiling. Because
they are runtime-mode controls, the public surface is context managers and
assignment rather than ``compile()`` keyword arguments — a settings change is
just a state change on a live module.

Two runtimes, one dispatch point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch-TensorRT has two engine runtimes: the Python runtime and the C++ runtime
(the ``torch.classes.tensorrt.Engine`` TorchBind class). Each carries its own
copy of the derived runtime state — there is no shared global.

``TorchTensorRTModule`` is the single point of contact. Assigning
``runtime_settings`` resolves the cache, dispatches the settings to whichever
runtime is attached, and stores the resolved value; which runtime is active is
a per-process fact (whether the C++ runtime library is loaded), and the module
adapts to it. Each engine holds its settings in a small shim, mirrored in
Python and C++, whose only responsibility is to own the settings, lazily derive
the live ``IRuntimeConfig`` from them, and keep all TensorRT-RTX feature gating
in one place so the rest of the runtime stays uniform.

Lazy execution context
^^^^^^^^^^^^^^^^^^^^^^^^

Both runtimes create the ``IExecutionContext`` lazily, on first use, and a
settings change *invalidates* it rather than rebuilding eagerly — the next use
recreates it with the new settings sampled in. In the common case an engine
therefore creates exactly one context across setup and first execution.

The exception is multi-device (NCCL) engines, which must bind their
communicator before any rank issues a collective, so they create the context
eagerly at setup. They consequently pay one extra context rebuild if settings
change afterwards. The trade-off is deliberate: cross-rank correctness outweighs
the extra create.

The runtime cache
^^^^^^^^^^^^^^^^^

The runtime kernel cache is the one piece of runtime state with non-trivial
structure, because it can be **shared across engines** — one cache can back
several engines under a single ``runtime_cache([...], path)`` block — and must
be reachable from **both runtimes**.

``RuntimeCache`` is a thin user-facing facade over a single inner handle. The
inner handle is the C++ TorchBind handle when the C++ runtime is loaded and a
pure-Python equivalent otherwise; both expose the same interface, so the facade
forwards to either without branching. When the handle is C++-backed it *is* the
same object the C++ engine holds — one refcounted instance visible from both
languages, which is what lets a single cache be shared across the boundary.

Materialization is deferred: the underlying TensorRT cache is created lazily, on
the same first-use path as the execution context, and any bytes loaded from disk
beforehand are held and drained in once it exists. This warm-start path is what
lets a path-backed cache be primed from disk up front and still attach correctly
to an engine whose cache only materializes on first execution.

Lifetime and autosave contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Who owns a cache determines when it is persisted. There are three origins:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Origin
     - Owner
     - Saved when
   * - **Engine-implicit** — the module builds it from a path string in the settings
     - the module
     - automatically: when replaced by a new cache, on garbage collection, and at interpreter exit
   * - **Shared** — ``runtime_cache(...)`` attaches it to every engine in scope
     - the context manager
     - on block exit
   * - **User-built** — constructed directly by advanced callers
     - the caller
     - manually via ``save()`` (opt into automatic save with ``autosave_on_del=True``)

Engine-implicit and shared caches never require an explicit save; user-built
ones do unless autosave is opted in. Automatic saves are idempotent — the
several hooks that can fire for an engine-implicit cache coordinate so it is
written at most once.

Exception safety
^^^^^^^^^^^^^^^^

Cache persistence is best-effort and never surfaces as a user-visible failure. A
save that fails (full disk, lock timeout, permission denied) is logged and
swallowed, so assigning ``runtime_settings`` or exiting a ``runtime_cache``
block cannot raise because of a cache write.

The engine-implicit cache is saved **synchronously at the moment it is
replaced** — not deferred to garbage collection — so a failure in the
subsequent settings dispatch can never strand freshly generated kernels, and the
interpreter-exit save does not depend on finalizer ordering, which is unreliable
during shutdown.

Concurrency
^^^^^^^^^^^

Everything other than the cache is per-engine and runs under the engine's
existing execution lock, so the settings and the config shims need no new
synchronization. The shared cache handle is the sole structure designed to be
touched by multiple engines, and therefore the sole structure that carries its
own lock — used to serialize the one-time cache materialization when several
engines share a handle across threads. Coarse locking suffices because these are
cold, infrequent operations, not per-inference ones.

The scoped context managers are a separate matter: their snapshot/restore state
is not synchronized, so overlapping ``runtime_config`` blocks on the *same*
module from different threads are unsupported (see `Limitations`_).

CUDA graphs integration
^^^^^^^^^^^^^^^^^^^^^^^^

``cuda_graph_strategy`` selects TensorRT-RTX-native CUDA-graph capture, which
must be set *before* the graph is captured. Rather than leave that ordering to
the caller, it is exposed as a keyword on
``enable_cudagraphs(..., cuda_graph_strategy=...)``, which applies the strategy
for the duration of the cudagraphs block and restores it afterwards — turning an
ordering requirement into structure. (The dynamic-shapes strategy has no such
coupling and keeps a plain scoped setter.)

Limitations
-----------

* The scoped context managers are not reentrant across threads — overlapping
  ``runtime_config`` blocks on the same module from different threads are
  unsupported.
* The default per-user cache path is safe against corruption (it is file-locked)
  but not against lost updates under concurrent processes; for CI or sweeps,
  give each worker its own path or opt out.
* Multi-device (NCCL) engines pay one extra execution-context rebuild for a
  post-setup settings change (see `Lazy execution context`_).
* A shared cache stays attached to every listed engine for the whole
  ``runtime_cache([...])`` block; there is no mid-block detach.

Testing
-------

Unit tests live under ``tests/py/dynamo/runtime/``: ``test_000_runtime_cache.py``
(cache lifetime, persistence, and stream I/O), ``test_004_runtime_settings.py``
(the settings dataclass, the context managers, the lazy-context invariant, and
cudagraphs composition), and ``test_001_cuda_graph_strategy.py`` /
``test_001_dynamic_shapes_kernel_strategy.py`` (per-strategy behavior).
Model-level coverage is under ``tests/py/dynamo/models/``.

Migration from compile-time knobs
---------------------------------

These knobs were previously passed to ``compile()``; they now live on
``RuntimeSettings``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Before (compile-time)
     - After (runtime)
   * - ``compile(..., cuda_graph_strategy="whole_graph_capture")``
     - ``mod.runtime_settings = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")``
   * - ``compile(..., dynamic_shapes_kernel_specialization_strategy="eager")``
     - ``mod.runtime_settings = RuntimeSettings(dynamic_shapes_kernel_specialization_strategy="eager")``
   * - ``compile(..., runtime_cache_path="/p")``
     - ``mod.runtime_settings = RuntimeSettings(runtime_cache="/p")`` or ``with runtime_cache(mod, "/p"):``

Related
-------

* :ref:`execution` — the runtime module architecture.
* :ref:`engine_caching_design` — the build-time engine cache, distinct from the runtime kernel cache described here.
* :doc:`/user_guide/runtime_performance/runtime_settings` — public API and usage.
