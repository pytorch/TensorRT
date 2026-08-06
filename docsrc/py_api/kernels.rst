.. _torch_tensorrt_kernels_py:

torch_tensorrt.kernels
======================

.. currentmodule:: torch_tensorrt.kernels

.. automodule:: torch_tensorrt.kernels

.. note::

   This module is **experimental**.  It requires ``cuda-python`` at runtime
   and TensorRT ``>=10.7.0`` (and not ``10.14.x``) for Quick Deployable
   Plugin (QDP) support.  Install ``cuda-python`` with ``pip install
   cuda-python``.

Overview
--------

The ``kernels`` module registers custom kernels — CUDA C++ compiled with
NVRTC, or cuTile — as TensorRT Quick Deployable Plugins. Tensor-only
declarative kernels use Ahead-of-Time (AOT) plugin launches when
available; kernels with ``ScalarInput`` compile through TensorRT's QDP JIT
path because QDP AOT extra arguments currently support symbolic integer
expressions, not arbitrary runtime floats.

Three entry points share one registration funnel:

* :func:`cuda_kernel_op` handles both the declarative case (drive
  everything from a :class:`KernelSpec` dataclass) and the override case
  (supply ``meta_fn`` / ``eager_fn`` / ``aot_fn`` / ``schema`` keyword
  arguments when the declarative DSL doesn't cover your kernel).
* :func:`ptx_op` registers kernels that are already compiled to PTX bytes.
* :func:`cutile_op` registers a ``@ct.kernel`` cuTile program, compiling it
  to PTX and deriving the AOT launch for you.

Entry points
------------

.. autofunction:: cuda_kernel_op

KernelSpec DSL
^^^^^^^^^^^^^^

.. autoclass:: KernelSpec
   :members:

.. autoclass:: InputDecl
   :members:

.. autoclass:: ScalarInput
   :members:

``ScalarInput`` values are represented as TensorRT plugin attributes during
compilation and are forwarded by value to the CUDA kernel.  Tensor-only
``cuda_kernel_op`` registrations use AOT plugin launches; registrations with
``ScalarInput`` use QDP JIT plugin execution so scalar floats / ints / bools can
be passed correctly.

.. autoclass:: OutputDecl
   :members:

Shape relations
"""""""""""""""

.. autoclass:: SameAs
   :members:

.. autoclass:: ReduceDims
   :members:

Extra scalar args
"""""""""""""""""

Extras are passed to the kernel between the input and output pointer
lists in :class:`KernelSpec` order.

.. autoclass:: Numel
   :members:

.. autoclass:: DimSize
   :members:

Launch geometry
"""""""""""""""

.. autoclass:: Elementwise
   :members:

.. autoclass:: Reduction
   :members:

.. autoclass:: Custom
   :members:

Override path
^^^^^^^^^^^^^

Pass any of the optional keyword arguments to :func:`cuda_kernel_op` to
bypass the corresponding auto-derivation:

* ``meta_fn`` — fake/meta impl: shape + dtype inference for tracing.
  When supplied, ``spec.outputs`` may be omitted.
* ``eager_fn`` — CUDA device impl invoked when the op runs in PyTorch
  eager. Same positional signature as ``meta_fn``.
* ``aot_fn`` — TensorRT AOT impl with signature
  ``(inputs, outputs, tactic) -> (KernelLaunchParams, SymExprs | None)``.
  When both ``eager_fn`` and ``aot_fn`` are supplied, ``spec.geometry``
  may be omitted.
* ``schema`` — explicit Torch schema (for example
  ``"(Tensor x, float alpha) -> Tensor"``). Falls back to deriving from
  ``spec.inputs`` / ``spec.outputs`` if both are present, else to
  inferring from ``meta_fn`` type hints.

Use the override path for shape-changing kernels, multi-output kernels,
or anything that doesn't fit the Elementwise / Reduction conventions.

Pre-compiled PTX entry point
----------------------------

.. autofunction:: ptx_op

cuTile entry point
------------------

.. autofunction:: cutile_op

:func:`cutile_op` is the cuTile analogue of :func:`cuda_kernel_op`.  It
compiles the kernel once with ``cuda.tile.compilation.export_kernel``, then
registers the PyTorch custom op, the TRT plugin descriptor, the AOT impl
embedding the PTX, and the Torch-TensorRT converter::

    import cuda.tile as ct
    import tensorrt.plugin as trtp
    import torch
    import torch_tensorrt.kernels as ttk

    TILE = 128

    @ct.kernel
    def add_one_kernel(x, out, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        ct.store(out, index=(pid,),
                 tile=ct.load(x, index=(pid,), shape=(tile_size,)) + 1.0)

    def add_one_meta(X: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(X)

    ttk.cutile_op(
        "my::add_one",
        kernel=add_one_kernel,
        signature={"x": "fp32", "out": "fp32"},
        meta_fn=add_one_meta,
        grid=lambda inputs, outputs: (
            trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
        ),
        constants={"tile_size": TILE},
    )

``signature`` lists the kernel's array parameters in declaration order —
inputs first, then outputs — mapped to their element type: a
:class:`torch.dtype`, a dtype name like ``"float32"``, or a short alias
like ``"fp32"``.  ``constants`` supplies the ``ct.Constant`` values baked
into the compiled symbol.  ``grid`` receives ``trtp.TensorDesc`` objects, so use
``.shape_expr`` to stay symbolic and keep one engine valid across shapes.

Because the launch is built from symbolic shape expressions, ``cutile_op``
supports dynamic shapes by default.  Pass ``eager_fn`` to also give the op
a CUDA implementation outside TensorRT, or ``aot_fn`` to replace the
derived launch entirely.  Threads-per-block comes from the ``.reqntid`` the
compiled kernel declares — cuTile vectorizes, so this is often below the
tile size, and it is a hard requirement rather than a hint.

.. note::

   cuTile groups each array's parameters as ``(ptr, extents..., strides...)``
   in kernel-declaration order, which is *not* the
   ``(input_ptrs..., extra_args..., output_ptrs...)`` order TensorRT's AOT
   launcher uses.  ``cutile_op`` permutes the compiled PTX's ``.entry``
   parameter list and supplies the matching extents and strides as AOT extra
   arguments.  A mismatch here does not fail loudly — the kernel would read
   whatever TensorRT placed in each slot — so the parameter count of the
   compiled PTX is checked against the signature at registration time and a
   disagreement raises :class:`RuntimeError`.

   ``ndim`` (default 1) is the rank each array is compiled for; a rank-1
   array's extent is the tensor's element count, which is what a kernel
   written against a flattened view expects and what lets one registration
   accept any input shape.  Pass ``ndim=`` or a ``"<dtype>[rank]"`` signature
   entry for kernels that index multi-dimensional tiles.

   A ``cutile_op`` registration compiles a single PTX for the given dtypes
   and ``constants``.  Inputs or outputs whose dtypes differ from the
   compiled ones are detected during conversion: the op is left out of the
   engine and runs in PyTorch, with a warning naming the mismatch.  Register
   a second op if you need a second dtype — multi-config autotuning is not
   yet supported.

   ``tileiras``, the cuTile compiler, ships with ``cuda-tile`` but is not on
   ``PATH`` by default; add the package's bin directory (e.g.
   ``<site-packages>/nvidia/cu13/bin``) before registering cuTile kernels.

   ``tileiras`` emits the PTX ISA of the toolkit it was built against, which
   can be newer than the installed driver loads.  Because TensorRT loads
   embedded PTX lazily, that would surface much later as an opaque
   ``onShapeChange status -1`` from the engine, so :func:`cutile_op` offers the
   compiled PTX to the driver at registration and raises if it is refused,
   naming the highest ISA the driver does accept.  The mismatch is reported
   rather than patched: lowering the ``.version`` header is a text substitution
   over a body compiled for a different ISA, so whether it survives depends on
   which instructions that body happens to contain.  Align the driver with the
   cuda-tile toolchain, or — having established a lower ISA is safe for your
   kernel — pass ``max_ptx_version=`` to set the header explicitly.

Kernel signature convention
---------------------------

All entry points assume the kernel takes its arguments in the fixed
order::

    (input_ptrs..., extras..., output_ptrs...)

This matches the order TensorRT passes tensor pointers and AOT extra
arguments.  In a CUDA C++ ``__global__`` kernel, pointers are ``void*``
cast to the appropriate element type; a cuTile kernel declares its arrays
in ``(inputs..., outputs...)`` order and ``cutile_op`` rewrites the
compiled PTX into the layout above.  Extras follow the order declared in
:attr:`KernelSpec.extras` for the declarative path, the extents and strides
:func:`cutile_op` derives from the signature, or the order your ``aot_fn``
builds for the override path.

Error behavior
--------------

:func:`cuda_kernel_op` validates the :class:`KernelSpec` at registration
time and raises :class:`ValueError` for the common authoring mistakes:

- Empty or duplicate-named ``inputs`` / ``outputs``.
- ``ReduceDims(input_idx=...)`` or ``SameAs(input_idx=...)`` where the
  reference is an out-of-range integer or a name that is not a tensor input.
  Both forms are accepted: an integer position into the tensor-only input
  list, or the input ``name`` (preferred — survives input reordering).
- ``Numel`` / ``DimSize`` referencing a name that is not an input.
- ``dtype_from`` pointing at an unknown input.
- ``Elementwise(layout='flat')`` with a multi-dimensional block tuple.
- Invalid block sizes, ``block_size`` in :class:`Reduction`, or a
  non-callable :attr:`Custom.fn`.
- A DSL field is missing and the corresponding override keyword argument
  was not supplied (e.g. ``outputs`` omitted without a ``meta_fn``).

Shape-dependent errors — for example
``Elementwise(layout='nd', block=(16, 16))`` invoked against a 1-D
output — are raised at launch time in a clear ``ValueError`` because
the offending ranks are only known when concrete tensors arrive.
