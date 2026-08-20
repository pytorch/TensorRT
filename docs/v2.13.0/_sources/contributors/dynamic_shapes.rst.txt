.. _dynamic_shapes_design:

Dynamic Shape Support
======================

.. note::

   This page documents the design for dynamic shape support in Torch-TensorRT.
   Original design discussions:
   `RFC #2014 <https://github.com/pytorch/TensorRT/discussions/2014>`_,
   `RFC #2409 <https://github.com/pytorch/TensorRT/discussions/2409>`_,
   `RFC #2634 <https://github.com/pytorch/TensorRT/discussions/2634>`_.

Goal
----

Support models whose tensor shapes vary at inference time (e.g. variable batch
size, variable sequence length) without recompiling the engine for each new shape.
TensorRT's **optimization profile** mechanism handles this: each engine is built
with a ``(min, opt, max)`` shape range for every dynamic dimension, and TensorRT
auto-tunes kernels for the ``opt`` shape while guaranteeing correctness for any
shape in ``[min, max]``.

User API
---------

``torch.export`` (AOT) Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Provide explicit ``dynamic_shapes`` annotations and use ``torch_tensorrt.Input``
with ``min_shape``/``opt_shape``/``max_shape``:

.. code-block:: python

    import torch
    import torch_tensorrt

    batch = torch.export.Dim("batch", min=1, max=8)
    exp_program = torch.export.export(
        model, inputs, dynamic_shapes={"x": {0: batch}}
    )

    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
            )
        ],
    )

``torch.compile`` (JIT) Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``dynamic=True`` to ``torch.compile``. Dynamo captures guards on input
shapes; if a guard fails the subgraph is retraced and a new TRT engine is built
for the new shape configuration:

.. code-block:: python

    optimized_model = torch.compile(
        model, backend="tensorrt", dynamic=True
    )
    # First call compiles for the observed shape; new shapes trigger recompile
    out = optimized_model(x)

Internal Implementation
------------------------

Shape Propagation
^^^^^^^^^^^^^^^^^

Dynamic dimensions are tracked as **SymPy symbolic expressions** (e.g. ``s0``,
``2 * s0 + 1``) throughout the FX graph. During conversion the symbolic
expressions are evaluated against the ``(min, opt, max)`` profile to derive
concrete integer bounds for each TRT layer's input/output sizes. This allows
shape inference to run at compile time without re-executing the model with real
data.

Optimization Profile Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``torch_tensorrt.Input`` with ``min/opt/max_shape`` is converted into a TRT
``IOptimizationProfile``. For multi-input models each profile covers all inputs
simultaneously. The builder selects the best kernels for the ``opt`` shape while
retaining correctness for any shape in the range.

SymInt Preservation
^^^^^^^^^^^^^^^^^^^^

When complex graph rewrites or lowering passes add new placeholder nodes (e.g.
during complex tensor decomposition), the new node's ``meta["val"]`` must carry
the correct symbolic shape — concretely, any dynamic dimensions must remain as
``SymInt`` objects rather than being collapsed to a fixed integer. This is
critical so that the optimization profile is derived correctly.

Interaction with Engine Caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Engine hashing (see :ref:`engine_caching_design`) takes dynamic shapes into
account: the ``(min, opt, max)`` profile is included in the hash key. Two
compilations for the same model but different shape ranges produce different
cache entries.

Interaction with CUDAGraphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDAGraphs (see :ref:`cuda_graphs_design`) are fundamentally incompatible with
arbitrary dynamic inputs because they record fixed memory addresses. Two
strategies exist:

* Record a separate CUDA Graph for each distinct shape (stored in a dict keyed
  on input shape).
* Use a single CUDA Graph recorded for the ``opt`` shape, and fall back to eager
  execution for other shapes.

Related
-------

* :ref:`lowering` — SymPy symbolic shapes are preserved through lowering.
* :ref:`conversion` — optimization profiles are built during conversion.
* :ref:`engine_caching_design` — shape range is included in the cache key.
* `Example: compile_with_dynamic_inputs.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/compile_with_dynamic_inputs.py>`_
* `Example: save_dynamic_shapes_example.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/save_dynamic_shapes_example.py>`_
