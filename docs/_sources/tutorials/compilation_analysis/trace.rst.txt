.. _trace:

Tracing Models with ``torch_tensorrt.dynamo.trace``
=====================================================

:func:`torch_tensorrt.dynamo.trace` is a thin wrapper around
``torch.export.export`` that automatically injects Torch-TensorRT's operator
decompositions at export time, producing an ``ExportedProgram`` that is better
suited for TRT compilation than a vanilla export.

----

Why Use ``trace`` Instead of ``torch.export.export`` Directly?
---------------------------------------------------------------

Both paths produce a ``torch.export.ExportedProgram``. The difference is in the
decompositions applied:

* **``torch.export.export``** — applies a default set of decompositions, some of which
  produce composite ops (e.g. ``aten.linear``, ``aten.embedding``) that Torch-TensorRT
  must decompose again in its own lowering pass.

* **``torch_tensorrt.dynamo.trace``** — applies Torch-TensorRT's curated ATen
  decomposition set upfront, producing a graph closer to the Core ATen opset that the
  converter library expects. This can reduce lowering time and increase TRT coverage for
  certain model architectures.

For most models the difference is minor. Use ``trace`` when:

* You are already using ``torch_tensorrt`` at the export step and want a single API.
* Your model has ops that Torch-TensorRT's lowering pass handles better when decomposed
  at export time.
* You want the tracing step to validate Torch-TensorRT compatibility before compilation.

----

Basic Usage
-----------

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn(1, 3, 224, 224).cuda()]

    exp_program = torch_tensorrt.dynamo.trace(model, arg_inputs=inputs)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, arg_inputs=inputs)

----

Dynamic Shapes
--------------

Pass ``torch_tensorrt.Input`` objects to specify dynamic dimensions. ``trace``
extracts ``torch.export.Dim`` constraints from the ``min/opt/max`` shapes and
passes them to ``torch.export.export`` automatically:

.. code-block:: python

    from torch_tensorrt import Input

    dyn_inputs = [
        Input(
            min_shape=(1,  3, 224, 224),
            opt_shape=(8,  3, 224, 224),
            max_shape=(16, 3, 224, 224),
            dtype=torch.float32,
            name="x",
        )
    ]

    exp_program = torch_tensorrt.dynamo.trace(model, arg_inputs=dyn_inputs)
    trt_gm = torch_tensorrt.dynamo.compile(exp_program, arg_inputs=dyn_inputs)

Under the hood, ``trace`` calls ``get_dynamic_shapes_args`` / ``get_dynamic_shapes_kwargs``
to build the ``dynamic_shapes`` dict required by ``torch.export.export``, using
the ``Input``'s min/max range to construct a ``torch.export.Dim`` for each dynamic
axis. Axes where min == max are treated as static.

----

Keyword Inputs
--------------

.. code-block:: python

    # Model with a kwarg:  def forward(self, x, *, mask=None)
    exp_program = torch_tensorrt.dynamo.trace(
        model,
        arg_inputs=(torch.randn(1, 512).cuda(),),
        kwarg_inputs={"mask": torch.ones(1, 512, dtype=torch.bool).cuda()},
    )

Both ``arg_inputs`` and ``kwarg_inputs`` accept ``torch.Tensor`` (static) or
``torch_tensorrt.Input`` (dynamic) values.

----

Non-Strict Tracing
------------------

``trace`` uses ``strict=False`` by default when calling ``torch.export.export``.
Non-strict mode allows data-dependent control flow and Python-side tensor operations
that strict export would reject. If you need strict export semantics, pass
``strict=True``:

.. code-block:: python

    exp_program = torch_tensorrt.dynamo.trace(model, arg_inputs=inputs, strict=True)

----

``trace`` vs ``compile`` Entry Points
---------------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * -
     - ``torch_tensorrt.dynamo.trace`` then ``compile``
     - ``torch_tensorrt.dynamo.compile`` directly (``nn.Module`` input)
   * - ExportedProgram reuse
     - Yes — export once, compile multiple times with different settings
     - No — re-exports on every compile call
   * - Decompositions at export
     - Torch-TRT curated set
     - Applied during compile's lowering pass
   * - Inspect pre-compilation graph
     - Yes — inspect ``exp_program.graph_module``
     - No — graph only visible inside compile
   * - ``torch.compile`` JIT path
     - N/A
     - Supported via ``backend="tensorrt"``
