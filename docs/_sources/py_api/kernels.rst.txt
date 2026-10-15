.. _torch_tensorrt_kernels_py:

torch_tensorrt.kernels
======================

.. currentmodule:: torch_tensorrt.kernels

.. automodule:: torch_tensorrt.kernels

.. note::

   This module is **experimental**. All entry points require TensorRT
   ``>=10.7.0`` (and not ``10.14.x``) with Quick Deployable Plugin (QDP)
   support. CUDA C++ kernels require ``cuda-python`` and ``cuda-core``;
   :func:`triton_op` requires ``triton>=3.5.0`` while registering and compiling
   the kernel. cuTile kernels additionally require ``cuda-tile`` and
   ``tileiras`` as described below. From a source checkout, install all
   frontend packages with ``pip install --group kernels``. A serialized AOT
   engine embeds the resulting PTX and does not need Triton, cuTile, or the
   Python registration callback to execute.

Overview
--------

The ``kernels`` module registers custom kernels — CUDA C++ compiled with
NVRTC, Triton, or cuTile — as TensorRT Quick Deployable Plugins. Tensor-only
declarative kernels use Ahead-of-Time (AOT) plugin launches when
available; kernels with ``ScalarInput`` compile through TensorRT's QDP JIT
path because QDP AOT extra arguments currently support symbolic integer
expressions, not arbitrary runtime floats.

Four entry points share one registration funnel:

* :func:`cuda_kernel_op` handles both the declarative case (drive
  everything from a :class:`KernelSpec` dataclass) and the override case
  (supply ``meta_fn`` / ``eager_fn`` / ``aot_fn`` / ``schema`` keyword
  arguments when the declarative DSL doesn't cover your kernel).
* :func:`ptx_op` registers kernels that are already compiled to PTX bytes.
* :func:`triton_op` registers a ``@triton.jit`` kernel, compiling it to
  PTX and deriving the AOT launch for you.
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
be passed correctly. A user ``aot_fn`` cannot currently be combined with a
scalar Torch schema because the QDP AOT callback receives only tensor
descriptors; the registration rejects that combination instead of dropping the
scalar values.

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

QDP schemas must declare every input positionally. Keyword-only Tensor or
scalar inputs are rejected because the generated TensorRT converter binds the
plugin ABI in schema order.

Use the override path for shape-changing kernels, multi-output kernels,
or anything that doesn't fit the Elementwise / Reduction conventions. AOT
override registrations must use Tensor-only schemas. For ``ScalarInput``, omit
``aot_fn`` and provide launch geometry so the registration selects the QDP JIT
path explicitly.

Pre-compiled PTX entry point
----------------------------

.. autofunction:: ptx_op

``ptx_op`` validates that ``ptx`` is non-empty UTF-8 text containing the
requested ``.entry`` symbol before it mutates the PyTorch or TensorRT global
registries. Its AOT path requires a Tensor-only Torch schema. Treat PTX and its
entry name as trusted build artifacts: this API validates the registration ABI,
not the behavior or memory safety of arbitrary device code.

Triton entry point
------------------

.. autofunction:: triton_op

:func:`triton_op` is the Triton analogue of :func:`cuda_kernel_op`. It
compiles the kernel ahead of time with ``triton.compile``, then registers the
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

    def add_one_meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.triton_op(
        "torch_tensorrt_example::triton_add_one",
        kernel=add_one_kernel,
        signature={"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"},
        constexprs={"BLOCK_SIZE": 256},
        grid=lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), 256),),
        meta_fn=add_one_meta,
        extra_args_fn=lambda inputs, outputs: [
            trtp.SymInt32(inputs[0].shape_expr.numel())
        ],
    )

This entry point requires Triton ``>=3.5.0``. That is the first release whose
NVIDIA compiler metadata exposes the complete scratch-memory and launch ABI
needed to validate that a compiled kernel can be launched safely through
TensorRT QDP. Older releases are rejected before compilation with an upgrade
command instead of producing a generic missing-compiler-metadata error after
an expensive compile.

``signature`` lists the kernel's non-constexpr parameters in declaration
order (pointers as an exact ``*<dtype>`` spelling, scalars as the bare dtype) and
``constexprs`` supplies the ``tl.constexpr`` values baked into the PTX.
``grid`` and ``extra_args_fn`` receive ``trtp.TensorDesc`` objects, so use
``.shape_expr`` to stay symbolic and keep one engine valid across shapes.
The signature keys must exactly match the kernel declaration after its
``tl.constexpr`` parameters are removed. The AOT ABI currently supports only
``i32`` scalar parameters; ``extra_args_fn`` must return exactly one ``int`` or
``trtp.SymInt32`` value for each scalar entry, and concrete integers must fit
the signed 32-bit range. The Torch custom-op schema itself must be Tensor-only:
these kernel scalars are launch metadata derived from ``TensorDesc`` shapes,
not user-facing Torch scalar attributes.

``triton_op`` declares dynamic-shape support by default, but one engine remains
valid across shapes only when ``meta_fn``, ``grid``, and ``extra_args_fn`` keep
their shape calculations symbolic. Pass ``eager_fn`` to also give the op a CUDA
implementation outside TensorRT. ``num_warps`` and ``num_stages`` are forwarded
to ``triton.compile``; ``num_warps`` also sets the launch's threads-per-block.
``num_warps`` must be a power of two from 1 to 32 and ``num_stages`` must be a
positive integer. Invalid callbacks, names, and options are rejected before
compilation. Duplicate operator names are also checked before compilation and
rechecked under the registration lock.

Registration compiles for the current CUDA device. For a different device,
register inside ``with torch.cuda.device(device):``. The compiled artifact
records its target, and conversion and AOT construction reject a different GPU
architecture. Use a separate operator name for each architecture. Triton's
existing cache handles repeated compilation of the same target/configuration;
Torch-TensorRT does not maintain a second persistent compilation cache.

The calling convention is enforced at registration time, because violating
it does not fail loudly — the kernel would read whatever TensorRT happened
to place in those argument slots.  :func:`triton_op` raises
:class:`ValueError` if ``signature`` begins or ends with a scalar,
interleaves scalars between pointers, disagrees with ``meta_fn``'s arity,
differs from the kernel declaration, contains an unknown pointer dtype or a
non-``i32`` scalar, or declares scalars without an ``extra_args_fn`` to supply
them. The launch is also rejected if ``extra_args_fn`` returns the wrong number
or type of values. Constant scalar values must fit in signed 32-bit integers,
including constants wrapped in ``SymInt32``. ``grid`` must return one to three
dimensions: X must be between 1 and 2**31 - 1, and Y and Z between 1 and 65535.
These limits are checked for ordinary integers and symbolic constants.

For expressions that depend on input shapes, ``grid`` and ``extra_args_fn``
must stay within those limits for every shape allowed by the input profile.
The adapter does not prove bounds for dynamic expressions. Empty inputs need
a positive launch grid and a kernel that masks every load and store.

.. note::

   A ``triton_op`` registration compiles a single PTX for the given
   ``signature`` and ``constexprs``. Supported pointer spellings are ``bf16``,
   ``fp16``, ``fp32``, ``fp8e4nv``, ``i1``, ``i8``, ``i32``, ``i64``, and
   ``u8``. Input and output pointer dtypes may differ from one another, but
   ``meta_fn`` must report each output dtype exactly as compiled. Inputs or
   outputs whose dtypes differ from the compiled ones, or whose FX tensor
   metadata is unavailable, are declined during conversion with a warning.
   They can run as a PyTorch fallback only if the surrounding partition and
   ``eager_fn`` allow it; otherwise compilation fails instead of installing an
   unchecked pointer binding. Register a second op for a second dtype —
   multi-config autotuning and dtype specialization are not yet supported.

   Triton emits PTX at the ISA version of its own bundled ``ptxas``, which
   can be newer than the installed CUDA driver accepts.  ``triton_op``
   attempts to detect the driver's maximum ISA and recompiles at that version
   when needed. If version discovery succeeds, this prevents embedding PTX
   newer than the active driver. The PTX is otherwise embedded exactly as
   Triton produced it.

   The AOT launcher currently supports ordinary, non-clustered CUDA launches.
   Registration fails closed if Triton's compiler metadata layout is incomplete
   or reports non-zero global/profile scratch, multiple CTAs per cluster, a
   non-32 warp size, cooperative launch, programmatic dependent launch, tensor
   memory, or tensor-descriptor metadata. Zero-sized scratch parameters are
   retained in Triton's PTX and rely on TensorRT padding unused arguments with
   null. This launch behavior has been validated with TensorRT 11.2 and Triton
   3.5.0 and 3.8.0. The general QDP API version requirement above does not imply
   that this Triton launch behavior has been validated on every earlier release.

Measuring and tuning configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/benchmark_triton_op.py`` compares a bounded set of block sizes and warp
counts. It checks correctness at each shape and reports compilation,
registration, engine-build, direct-Triton, and TensorRT execution timings as
JSON. Use a fresh Triton cache directory to measure cold compilation::

    TRITON_CACHE_DIR=$(mktemp -d) python tools/benchmark_triton_op.py \
        --blocks 128 256 512 --warps 4 8 --sizes 1024 262144 1048576 \
        --output triton-benchmark.json

The benchmark uses a fused elementwise kernel; repeat the measurements with
your workload before choosing a configuration. One configuration remains fixed
for each registered operator. No alignment or divisibility assumptions are
added automatically: a dynamic shape or an offset input pointer can invalidate
them. All embedded PTX is exactly Triton's compiler output, including its hidden
zero-sized scratch parameters. Compilation may select a compatible PTX ISA,
but no generated PTX text or parameter lists are rewritten.

The fast CI suite ``kernels-triton`` covers registration and GPU integration.
Nightly kernel tests also run against Triton 3.5.0 in an isolated subprocess.
To reproduce that compatibility run without replacing the installed Triton::

    python -m tests.ci.triton_compat 3.5.0

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

    def add_one_meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.cutile_op(
        "cutile_example::add_one",
        kernel=add_one_kernel,
        signature={"x": "fp32", "out": "fp32"},
        meta_fn=add_one_meta,
        grid=lambda inputs, outputs: (
            trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
        ),
        constants={"tile_size": TILE},
    )

Install the AOT API with ``pip install 'cuda-tile>=1.3,<2'`` and expose
``tileiras`` from a system CUDA Toolkit compatible with the installed
``cuda-tile`` through ``PATH`` or ``CUDA_HOME``. In environments without
separately pinned CUDA Toolkit packages, the ``cuda-tile[tileiras]`` extra can
provide the compiler instead; its CUDA package constraints must match the rest
of the environment.

``signature`` lists the kernel's array parameters in declaration order —
inputs first, then outputs — mapped to their element type: a
:class:`torch.dtype`, a dtype name like ``"float32"``, or a short alias
like ``"fp32"``.  ``constants`` supplies the ``ct.Constant`` values baked
into the compiled symbol.  ``grid`` receives ``trtp.TensorDesc`` objects, so use
``.shape_expr`` to stay symbolic and keep one engine valid across shapes.
Signature dtypes must have an exact TensorRT representation: ``float64`` and
``int16``, for example, are rejected before compilation. ``truncate_double``
cannot make an fp64 kernel compatible with float32 engine buffers. Actual
input and output descriptor dtypes are checked again before the AOT launch
callback, including when a custom ``aot_fn`` is supplied.
Input names and order must match the Torch schema inferred from ``meta_fn``;
this is how each TensorRT pointer is tied to its intended kernel parameter.

Because the launch is built from symbolic shape expressions, ``cutile_op``
supports dynamic shapes by default.  Pass ``eager_fn`` to also give the op
a CUDA implementation outside TensorRT, or ``aot_fn`` to replace the
derived launch entirely.  Threads-per-block comes from the ``.reqntid`` the
compiled kernel declares — cuTile vectorizes, so this is often below the
tile size, and it is a hard requirement rather than a hint.

A custom ``aot_fn`` must return the extents and strides described by
``signature`` in a ``trtp.SymIntExprs`` container, in input-then-output order.
``cutile_op`` checks the container, item type, and count, but the caller owns
the symbolic values and their order; a semantic mismatch can silently corrupt
the result. Block dimensions must be build-time constants and must exactly
match the compiled kernel's ``.reqntid`` when one is present. Caller-supplied
dynamic shared memory is unsupported and must be zero. These launch properties
are checked before TensorRT accepts the launch.

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

   The Torch operator schema must contain Tensor inputs and Tensor outputs
   only. Runtime scalar parameters cannot be forwarded by this AOT ABI; make
   them ``ct.Constant`` parameters and supply them through ``constants``.

   This frontend requires CUDA Toolkit **13.2 or newer** and a loaded CUDA
   driver supporting **PTX ISA 9.2 or newer** (CUDA driver API 13.2+). This
   requirement applies even when cross-compiling with ``arch_override``.
   The selected ``tileiras`` toolkit version must not exceed the CUDA version
   supported by the loaded driver. For example, a CUDA 13.2 driver requires
   a 13.2 compiler; a CUDA 13.4 driver permits compilers from 13.2 through 13.4.
   Keep ``tileiras``, ``ptxas``, and ``libnvvm`` on matching toolkit major/minor
   versions. The frontend logs the selected versions and toolkit guidance,
   and rejects incompatible combinations before compilation.

   This path extracts PTX from cuTile's CUBIN and gives that PTX to TensorRT
   QDP. Installing a newer toolkit does not upgrade the driver's PTX JIT.
   Native cuTile's R580 support therefore does not establish support for this
   frontend. Upgrade the driver or configure supported CUDA forward-compatibility
   libraries when the loaded driver API is older than 13.2.

   The exact compiled PTX is also loaded through the driver at registration
   and rejected if incompatible. ``max_ptx_version=`` must be at least ``92``;
   it does not bypass the driver/toolkit checks or make newer instructions
   compatible with an older driver.

   ``arch_override=`` may be used to cross-compile for another GPU. PTX for the
   current device is checked by loading it through the CUDA driver; a different
   target cannot be checked locally and must be validated on that target before
   deployment.

Kernel signature convention
---------------------------

The Triton and tensor-only precompiled AOT paths use this fixed order::

    (input_ptrs..., extras..., output_ptrs...)

This matches the order TensorRT passes tensor pointers and AOT extra
arguments. In a Triton kernel, pointers are the ``*<dtype>`` parameters
specified in ``signature`` and extras follow the order returned by
``extra_args_fn``; no PTX rewriting is needed. A cuTile kernel declares its
arrays in ``(inputs..., outputs...)`` order, and :func:`cutile_op` rewrites
the compiled PTX into the layout above, deriving extents and strides from
the signature as extra arguments.

A declarative CUDA C++ kernel receives every :attr:`KernelSpec.inputs` entry
in declaration order (tensor pointer or ``ScalarInput`` value), then
:attr:`KernelSpec.extras`, then output pointers. For :func:`ptx_op` and the
CUDA override path, ``aot_fn`` owns the launch and extra-argument order;
their Torch schemas must remain Tensor-only when AOT is selected.

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
