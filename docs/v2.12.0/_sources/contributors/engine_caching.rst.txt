.. _engine_caching_design:

Engine Caching
===============

.. note::

   This page documents the design for engine caching in Torch-TensorRT. The original
   design discussion is `RFC #2957 <https://github.com/pytorch/TensorRT/discussions/2957>`_.

Goal
----

Boost performance when calling ``torch.compile()`` or ``torch_tensorrt.compile()``
by reusing previously compiled TensorRT engines rather than recompiling the model
every time. Engine compilation (including kernel auto-tuning) can take minutes to
hours for large models; caching eliminates this overhead on subsequent runs.

High-Level Flow
----------------

After the partitioning phase, each TRT subgraph is hashed and looked up in the
cache before invoking the builder:

.. code-block:: text

    FX Graph
        │
        ▼
    Partition into TRT / PyTorch subgraphs
        │
        ▼  (per TRT subgraph)
    ┌──────────────────────────────────┐
    │  hash subgraph (architecture     │
    │  only — weights zeroed out)       │
    └───────────┬──────────────────────┘
                │
        ┌───────▼──────────┐
        │  cache hit?       │
        └───┬───────────┬──┘
           Yes           No
            │             │
            ▼             ▼
        load engine    build engine
        refit weights  save to cache
            │             │
            └──────┬───────┘
                   ▼
           serialized TRT engine

.. image:: https://github.com/pytorch/TensorRT/assets/29559374/f67aa8b3-ee43-4c46-a797-5da958e69a89
   :alt: Engine caching pipeline diagram

User API
---------

Engine caching is controlled by the ``cache_built_engines`` and
``reuse_cached_engines`` compilation settings:

.. code-block:: python

    import torch_tensorrt

    trt_gm = torch_tensorrt.compile(
        model,
        arg_inputs=inputs,
        cache_built_engines=True,   # save engines to disk after building
        reuse_cached_engines=True,  # load engines from disk on cache hit
    )

A higher-level wrapper, ``MutableTorchTensorRTModule``, enables engine caching
transparently alongside weight refit:

.. code-block:: python

    from torch_tensorrt.dynamo import MutableTorchTensorRTModule

    mutable = MutableTorchTensorRTModule(model, config=settings)
    # first call compiles and caches; subsequent calls reuse the cache

Design
-------

Graph Hashing
^^^^^^^^^^^^^

Two graphs are considered *isomorphic* if they share the same operator topology and
layer configuration. Weights are intentionally excluded — the cache key depends only
on architecture so that weight-updated variants of the same model still hit the cache.

Implementation:

1. All named parameters in the ``torch.fx.GraphModule`` are zeroed in-place.
2. PyTorch Inductor's ``FxGraphCachePickler`` hashes the resulting structure.

.. code-block:: python

    from torch._inductor.codecache import FxGraphCachePickler

    for name, param in gm.named_parameters():
        param.data.zero_()

    hash_val = FxGraphCachePickler.get_hash(gm)

Cache Operations
^^^^^^^^^^^^^^^^

The ``BaseEngineCache`` abstract class defines the interface:

* ``get_hash(gm)`` — produce a stable hash from the GraphModule structure.
* ``contains(hash)`` — check whether a serialized engine exists for this hash.
* ``save(hash, serialized_engine, input_specs, device_info)`` — persist an engine.
* ``load(hash)`` — retrieve a serialized engine; returns ``None`` on miss.

Two concrete implementations are provided:

* ``DiskEngineCache`` — stores engines as ``<cache_dir>/<hash>/engine.bin`` files
  on the local filesystem.  This is the default.
* ``MemoryEngineCache`` — stores engines in a Python ``dict`` keyed on hash; useful
  for testing and short-lived workloads.

Cache Eviction
^^^^^^^^^^^^^^

The ``DiskEngineCache`` uses a **Least Recently Used (LRU)** eviction strategy with
a configurable maximum cache directory size. When the limit is reached the least
recently accessed engine is removed first.

Weight Refit on Cache Hit
^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the hash ignores weights, a cache hit for a model with *updated* weights
requires re-applying the new weights to the loaded engine. This is done via the
:ref:`weight refit <weight_refit_design>` subsystem — the refit map constructed
during the original compilation is reused to copy new weight values into the cached
engine without rebuilding from scratch.

Cache Structure on Disk
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    /tmp/torch_tensorrt_engine_cache/   (default, configurable)
    └── <hash>/
        └── engine.bin                  (serialized TRT engine bytes)

Custom Cache Backends
^^^^^^^^^^^^^^^^^^^^^^

Users can supply their own cache backend by subclassing ``BaseEngineCache``:

.. code-block:: python

    from torch_tensorrt.dynamo import BaseEngineCache

    class MyS3Cache(BaseEngineCache):
        def save(self, hash, serialized_engine, ...):
            # upload to S3
            ...

        def load(self, hash):
            # download from S3 or return None
            ...

    trt_gm = torch_tensorrt.compile(
        model, arg_inputs=inputs,
        cache_built_engines=True,
        reuse_cached_engines=True,
        custom_engine_cache=MyS3Cache(),
    )

Related
-------

* :ref:`weight_refit_design` — weight refit is invoked on every cache hit.
* :ref:`execution` — the runtime module that executes compiled engines.
* `Examples: engine_caching_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/engine_caching_example.py>`_
* `Examples: engine_caching_bert_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/engine_caching_bert_example.py>`_
