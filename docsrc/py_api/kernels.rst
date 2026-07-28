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
NVRTC, or Triton — as TensorRT Quick Deployable Plugins. Tensor-only
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
* :func:`triton_op` registers a ``@triton.jit`` kernel, compiling it to
  PTX and deriving the AOT launch for you.

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

Triton entry point
------------------

.. autofunction:: triton_op

:func:`triton_op` is the Triton analogue of :func:`cuda_kernel_op`. It
compiles the kernel once with ``triton.compile``, then registers the
PyTorch custom op, the TRT plugin descriptor, the AOT impl embedding the
PTX, and the Torch-TensorRT converter — replacing the hand-written
``@trtp.aot_impl`` boilerplate in the :ref:`aot_plugin` example::

    import tensorrt.plugin as trtp
    import torch
    import triton
    import triton.language as tl
    import torch_tensorrt.kernels as ttk

    @triton.jit
    def add_one_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1, mask=mask)

    def add_one_meta(X: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(X)

    ttk.triton_op(
        "my::add_one",
        kernel=add_one_kernel,
        signature={"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"},
        constexprs={"BLOCK_SIZE": 256},
        grid=lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), 256),),
        meta_fn=add_one_meta,
        extra_args_fn=lambda inputs, outputs: [
            trtp.SymInt32(inputs[0].shape_expr.numel())
        ],
    )

``signature`` lists the kernel's non-constexpr parameters in declaration
order (pointers as ``*<dtype>``, scalars as the bare dtype) and
``constexprs`` supplies the ``tl.constexpr`` values baked into the PTX.
``grid`` and ``extra_args_fn`` receive ``trtp.TensorDesc`` objects, so use
``.shape_expr`` to stay symbolic and keep one engine valid across shapes.

Because the launch is built from symbolic shape expressions, ``triton_op``
supports dynamic shapes by default.  Pass ``eager_fn`` to also give the op
a CUDA implementation outside TensorRT, or ``aot_fn`` to replace the
derived launch entirely.

.. note::

   A ``triton_op`` registration compiles a single PTX for the given
   ``signature`` and ``constexprs``, so runtime input dtypes must match
   the compiled ones.  Multi-config autotuning and dtype specialization
   are not yet supported.

   Triton emits PTX at the ISA version of its own bundled ``ptxas``, which
   can be newer than the installed CUDA driver accepts.  ``triton_op``
   detects the driver's maximum ISA and recompiles at that version when
   needed, so the embedded PTX always loads.  It also strips Triton's
   trailing zero-sized scratch parameters, which TensorRT's AOT launcher
   does not supply; a kernel needing non-zero scratch cannot use the AOT
   QDP path and raises at registration time.

Kernel signature convention
---------------------------

All entry points assume the kernel takes its arguments in the fixed
order::

    (input_ptrs..., extras..., output_ptrs...)

This matches the order TensorRT passes tensor pointers and AOT extra
arguments, so no PTX rewriting is needed.  In a CUDA C++ ``__global__``
kernel, pointers are ``void*`` cast to the appropriate element type; in a
Triton kernel they are the ``*<dtype>`` parameters declared in
``signature``.  Extras follow the order declared in
:attr:`KernelSpec.extras` for the declarative path, the order
``extra_args_fn`` returns for :func:`triton_op`, or the order your
``aot_fn`` builds for the override path.

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
