.. _complex_tensors:

Complex Tensor Support
=======================

TensorRT does not natively support ``complex64`` or ``complex128`` tensors. Torch-TensorRT
handles them automatically via the ``complex_graph_rewrite`` lowering pass, which
rewrites complex-valued subgraphs into equivalent real-valued arithmetic before
compilation.

This page explains what the rewriter does, which patterns are supported, and what
limitations to be aware of when compiling models with complex inputs.

.. seealso::

   :doc:`../_rendered_examples/dynamo/torch_export_3d_rope` — a runnable
   end-to-end example compiling a video-transformer 3D RoPE attention block
   (CogVideoX / Wan / HunyuanVideo style) with dynamic T×H×W shapes.

----

How the Rewriter Works
----------------------

The ``complex_graph_rewrite`` pass runs as part of the standard lowering pipeline.
It:

1. **Detects** complex subgraphs by anchoring on ``view_as_real`` nodes and walking
   backward through the graph to find all upstream complex operations.
2. **Replaces** complex inputs with real-valued equivalents:
   - ``placeholder`` inputs of type ``complex64``/``complex128`` are replaced by new
     ``float32``/``float64`` placeholders with an appended trailing dimension of size 2
     (real and imaginary parts interleaved as ``(..., 2)``).
   - ``get_attr`` buffers that are complex are replaced by a new buffer produced by
     ``torch.stack([original.real, original.imag], dim=-1)``.
3. **Rewrites** complex multiply as explicit real arithmetic:
   ``(a+bi) * (c+di) = (ac - bd) + (ad + bc)i``
4. **Bypasses** ``view_as_real`` and ``view_as_complex`` nodes — they become
   identity-like operations after the rewrite and are erased from the graph.

The net result is a fully real-valued graph that TRT can compile natively.

----

Supported Patterns
------------------

The rewriter handles the following patterns inside a complex subgraph:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Pattern
     - How it is handled
   * - ``complex64`` / ``complex128`` input placeholder
     - Replaced by ``float32`` / ``float64`` placeholder with shape ``(..., 2)``
   * - ``complex64`` / ``complex128`` model buffer (``get_attr``)
     - Replaced by stacked real+imag buffer with shape ``(..., 2)``
   * - ``aten.mul.Tensor`` between complex tensors
     - Rewritten to ``(ac - bd) + (ad + bc)i`` real arithmetic
   * - ``aten.view_as_complex`` nodes
     - Erased (the input real tensor flows through unchanged)
   * - ``aten.view_as_real`` anchor nodes
     - Erased (the output is already real after the rewrite)
   * - ``aten.permute.default`` on complex tensors
     - Handled — the trailing ``2`` dimension is appended to the dims list

----

Usage
-----

No API changes are needed. The rewriter runs automatically whenever the exported graph
contains complex-valued nodes:

.. code-block:: python

    import torch
    import torch_tensorrt

    class RoPEModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            # x and freqs are real; view_as_complex converts to complex for mul
            x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            x_rotated = x_complex * freqs
            return torch.view_as_real(x_rotated).flatten(2)

    model = RoPEModel().eval().cuda()
    x = torch.randn(1, 16, 64).cuda()
    freqs = torch.randn(1, 16, 32, dtype=torch.complex64).cuda()

    exp_program = torch.export.export(model, (x, freqs))
    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=[x, freqs],
        use_explicit_typing=True,  # enabled_precisions deprecated
        min_block_size=1,
    )

    output = trt_gm(x, freqs)

The compiler detects the ``view_as_real`` node, walks the complex subgraph backward,
replaces the ``complex64`` input ``freqs`` with a ``float32`` placeholder of shape
``(1, 16, 32, 2)``, and rewrites the multiply.

.. _complex_inputs:

**Passing complex inputs at runtime:**

When the compiled model has complex input placeholders, pass the complex tensor directly.
The Torch-TensorRT runtime modules automatically call ``torch.view_as_real`` on complex
inputs before handing them to the TRT engine:

.. code-block:: python

    # freqs is still complex64 at call time — the runtime handles the conversion
    output = trt_gm(x, freqs)

----

``truncate_double``
--------------------

By default, ``complex128`` inputs are lowered to ``float64`` (two doubles). Set
``truncate_double=True`` in :ref:`compilation_settings` to truncate them to
``float32`` instead:

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=inputs,
        truncate_double=True,   # complex128 → float32 (saves memory, loses precision)
    )

----

Limitations
-----------

* **Only ``view_as_real``-anchored subgraphs** are detected. If your model uses
  complex arithmetic without ``view_as_real`` as the output boundary (e.g. a complex
  output tensor is returned directly), the subgraph will not be detected and the
  compilation will fail.

* **``view_as_complex`` must be paired with ``view_as_real``** in the same subgraph.
  Standalone ``view_as_complex`` nodes outside the detected subgraph are not handled.

* **No support for complex convolution or complex batch norm** — only element-wise
  ``mul.Tensor`` is rewritten. Complex convolution patterns must be decomposed manually
  into real arithmetic before compilation.

* **``complex128`` on GPU** requires ``float64`` support in TRT. Most consumer GPUs
  have limited ``float64`` throughput; use ``truncate_double=True`` for
  performance-critical workloads.

* **Parameters shaped ``(d, 2)`` (intentional, not complex)** — if a real parameter
  happens to have a trailing dimension of 2 and is consumed by a node that the
  detector considers "complex", it will not be mistakenly rewritten because the
  parameter's dtype is real. The rewriter only rewrites nodes whose ``meta["val"].dtype``
  is ``complex64`` or ``complex128``.
