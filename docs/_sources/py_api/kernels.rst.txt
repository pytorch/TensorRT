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

The ``kernels`` module registers NVRTC-compiled CUDA C++ kernels as
TensorRT Quick Deployable Plugins. Tensor-only declarative kernels use
Ahead-of-Time (AOT) plugin launches when available; kernels with
``ScalarInput`` compile through TensorRT's QDP JIT path because QDP AOT
extra arguments currently support symbolic integer expressions, not
arbitrary runtime floats.

A single function — :func:`cuda_kernel_op` — handles both the declarative
case (drive everything from a :class:`KernelSpec` dataclass) and the
override case (supply ``meta_fn`` / ``eager_fn`` / ``aot_fn`` / ``schema``
keyword arguments when the declarative DSL doesn't cover your kernel).
:func:`ptx_op` is a parallel entry point for kernels that are already
compiled to PTX bytes.

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

Kernel signature convention
---------------------------

All entry points assume the ``__global__`` kernel takes its arguments in
the fixed order::

    (input_ptrs..., extras..., output_ptrs...)

Pointers are ``void*`` cast to the appropriate element type.  Extras
follow the order declared in :attr:`KernelSpec.extras` for the
declarative path, or the order your ``aot_fn`` builds for the override
path.

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
