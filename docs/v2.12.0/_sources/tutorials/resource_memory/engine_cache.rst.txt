.. _engine_cache:

Engine Caching
==============

TRT engine compilation is the most expensive step in the Torch-TensorRT workflow. For
repeated compilations of the same model (e.g., after process restart, during
hyperparameter search, or in CI), the engine cache eliminates redundant builds by
persisting compiled engines to disk and reloading them on a cache hit.

----

Enabling the Cache
------------------

Pass ``cache_built_engines=True`` and ``reuse_cached_engines=True`` to
:func:`~torch_tensorrt.dynamo.compile`:

.. code-block:: python

    import torch
    import torch_tensorrt

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        cache_built_engines=True,
        reuse_cached_engines=True,
    )

By default the cache lives at
``/tmp/torch_tensorrt_engine_cache/`` with a 5 GB size limit.

Customize the cache location and size:

.. code-block:: python

    from torch_tensorrt.dynamo._engine_cache import DiskEngineCache

    my_cache = DiskEngineCache(
        engine_cache_dir="/data/trt_cache",
        engine_cache_size=20 * 1024**3,  # 20 GB
    )

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache=my_cache,
    )

----

What Gets Cached
-----------------

Each TRT subgraph is cached independently under a SHA-256 hash derived from three
components:

1. **Graph structure** — a canonicalized string of node ops and targets (placeholder
   names are normalized so renaming inputs does not bust the cache).
2. **Input specs** — the ``min/opt/max`` shapes and dtypes of each input tensor.
3. **Engine-invariant settings** — the subset of
   :class:`~torch_tensorrt.dynamo.CompilationSettings` that affect the compiled engine
   (see :ref:`engine-invariant-settings`). Settings like ``debug`` or ``dryrun`` do not
   affect the cache key.

Each cache entry stores:

* The serialized TRT engine bytes.
* Input and output tensor names.
* The original input specs (for verification on reload).
* The weight name map (for refit support).
* Whether the engine requires an output allocator (data-dependent shape ops).

Cache entries are stored as ``{cache_dir}/{hash}/blob.bin``.

----

Cache Invalidation
-------------------

The cache is **automatically invalidated** when any engine-invariant setting changes.
The following changes always require a cache miss (engine rebuild):

* ``enabled_precisions``
* ``max_aux_streams``
* ``version_compatible`` / ``hardware_compatible``
* ``optimization_level``
* ``disable_tf32`` / ``sparse_weights``
* ``engine_capability``
* ``immutable_weights`` / ``refit_identical_engine_weights`` / ``enable_weight_streaming``
* ``tiling_optimization_level`` / ``l2_limit_for_tiling``
* All ``autocast_*`` settings

Changes to ``min_block_size``, ``torch_executed_ops``, ``debug``, ``dryrun``,
``pass_through_build_failures``, etc. do **not** invalidate cached engines.

----

LRU Eviction
------------

When a new engine would exceed the configured ``engine_cache_size``,
``DiskEngineCache`` evicts the least-recently-used entries (based on file modification
time) until enough space is available. An engine larger than the total cache size is
silently not cached (a warning is logged).

----

Timing Cache
-------------

Separate from the engine cache, TRT maintains a **timing cache** that records kernel
benchmark results. This speeds up subsequent engine builds for similar subgraphs even on
a cold engine cache, because TRT can skip re-benchmarking known-fast kernels.

The timing cache is always active and persisted at ``timing_cache_path``:

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        timing_cache_path="/data/trt_cache/timing_cache.bin",
    )

The default path is
``/tmp/torch_tensorrt_engine_cache/timing_cache.bin``.

.. note::

   The timing cache is **not used with TensorRT-RTX**, which does not perform
   autotuning. For TensorRT-RTX, see the *Runtime Cache* section below.

Runtime Cache (TensorRT-RTX)
-----------------------------

TensorRT-RTX uses JIT compilation at inference time. The **runtime cache** stores
these compilation results so that kernels and execution graphs are not recompiled
on subsequent runs. This is analogous to the timing cache but operates at inference
time rather than build time.

The runtime cache is automatically created when using TensorRT-RTX and can be
persisted to disk via ``runtime_cache_path``:

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        runtime_cache_path="/data/trt_cache/runtime_cache.bin",
        use_python_runtime=True,
    )

The default path is
``/tmp/torch_tensorrt_engine_cache/runtime_cache.bin``.

The cache is saved to disk when the module is destroyed (garbage collected) and
loaded on subsequent compilations with the same path. File locking is used to
prevent corruption when multiple processes share the same cache file.

----

Custom Cache Backends
-----------------------

To store engines in a location other than the local disk (e.g., a shared object store,
a database), implement the ``BaseEngineCache`` interface:

.. code-block:: python

    from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
    from typing import Optional

    class S3EngineCache(BaseEngineCache):
        def __init__(self, bucket: str, prefix: str = "trt_engines/"):
            import boto3
            self.s3 = boto3.client("s3")
            self.bucket = bucket
            self.prefix = prefix

        def save(self, hash: str, blob: bytes) -> None:
            key = f"{self.prefix}{hash}/blob.bin"
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=blob)

        def load(self, hash: str) -> Optional[bytes]:
            key = f"{self.prefix}{hash}/blob.bin"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                return resp["Body"].read()
            except self.s3.exceptions.NoSuchKey:
                return None

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache=S3EngineCache("my-model-cache-bucket"),
    )

The two methods you must implement:

``save(hash: str, blob: bytes) -> None``
    Persist the packed blob (already serialized by ``BaseEngineCache.pack()``) under
    the given hash key.

``load(hash: str) -> Optional[bytes]``
    Return the packed blob for the given hash, or ``None`` on a cache miss.
    Returning ``None`` causes a normal engine build and subsequent ``save`` call.

The base class provides ``get_hash()``, ``pack()``/``unpack()``, ``insert()``, and
``check()`` — do not override these unless you understand the serialization format.

----

Weightless Engines (TRT ≥ 10.14)
-----------------------------------

On TRT 10.14 and later, engines can be serialized **without weights** using TRT's
``INCLUDE_REFIT`` flag. This significantly reduces cache storage for models where the
architecture is shared across many weight variants (e.g., different fine-tuned
checkpoints of the same base model):

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        strip_engine_weights=True,
        cache_built_engines=True,
        reuse_cached_engines=True,
        immutable_weights=False,
    )

On a cache hit the weightless engine is loaded and refitted with the current weights
before inference. The ``strip_engine_weights`` setting is part of the engine-invariant
set on TRT < 10.14 (different cache key), but handled automatically by TRT itself on
10.14+.
