.. _impl_subpackage:

The ``impl/`` Building-Block Library
======================================

``torch_tensorrt.dynamo.conversion.impl`` is a library of pre-built TRT layer
primitives that converter authors can compose rather than writing raw
``trt.INetworkDefinition`` API calls. Each module handles shape broadcasting,
type coercion, and naming boilerplate internally.

All modules are available directly under ``impl``:

.. code-block:: python

    from torch_tensorrt.dynamo.conversion import impl

    # example: compose a GeLU approximation
    x_sq = impl.elementwise.mul(ctx, target, name="x_sq", source_ir=SourceIR.ATEN,
                                lhs_val=x, rhs_val=x)

The standard call signature for most ``impl`` functions is:

.. code-block:: python

    def op(
        ctx: ConversionContext,
        target: Target,
        source_ir: Optional[SourceIR],
        name: str,
        # ... op-specific args ...
    ) -> TRTTensor:

``source_ir`` is the :ref:`SourceIR <source_ir>` enum value that identifies which
IR level this call originated from (used for debug naming). Pass
``SourceIR.ATEN`` from ATen converters.

----

Module Reference
-----------------

activation
^^^^^^^^^^

**``impl.activation``** — Standard activation functions.

Functions: ``relu``, ``sigmoid``, ``tanh``, ``leaky_relu``, ``elu``, ``selu``,
``softsign``, ``softplus``, ``gelu``, ``hard_sigmoid``, ``hard_swish``,
``prelu``, ``hardshrink``, ``softshrink``, ``tanhshrink``.

Each function wraps ``ctx.net.add_activation()`` with the appropriate
``trt.ActivationType`` and handles dynamic range propagation for INT8.

.. code-block:: python

    out = impl.activation.relu(ctx, target, SourceIR.ATEN, name, input_val)

addmm
^^^^^

**``impl.addmm``** — Fused add + matrix multiply (``beta * input + alpha * (mat1 @ mat2)``).
Uses ``add_matrix_multiply`` + ``add_elementwise`` with scale factors.

arange
^^^^^^

**``impl.arange``** — Generates a 1-D range tensor (``torch.arange`` semantics).
Implemented via TRT's ``IFillLayer`` with ``LINSPACE`` fill.

cast
^^^^

**``impl.cast``** — Explicit dtype casts. Wraps ``IIdentityLayer`` with
``set_output_type()``. Used internally by the autocast pass and type-enforcement
decorators.

cat
^^^

**``impl.cat``** — Tensor concatenation along a given dimension.
Wraps ``ctx.net.add_concatenation()``.

conv / deconv
^^^^^^^^^^^^^

**``impl.conv``** / **``impl.deconv``** — Convolution and transposed convolution.
Wraps ``add_convolution_nd`` / ``add_deconvolution_nd``, handles 1D/2D/3D,
padding modes, dilation, groups, and optional bias.

.. code-block:: python

    out = impl.conv.convolution(
        ctx, target, SourceIR.ATEN, name,
        input=x, weight=w, bias=b,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        transposed=False, output_padding=output_padding,
    )

dynamic_block_quantize
^^^^^^^^^^^^^^^^^^^^^^^

**``impl.dynamic_block_quantize``** — Block-wise dynamic quantization helpers for
FP8 and FP4 workflows. Wraps TRT's quantization layers.

elementwise
^^^^^^^^^^^

**``impl.elementwise``** — Element-wise binary and unary operations.

Binary ops: ``add``, ``sub``, ``mul``, ``div``, ``pow``, ``floor_div``,
``trunc_div``, ``fmod``, ``logical_and``, ``logical_or``, ``logical_xor``,
``bitwise_and``, ``bitwise_or``, ``bitwise_xor``, ``eq``, ``ne``, ``gt``,
``ge``, ``lt``, ``le``, ``max``, ``min``.

Unary ops: ``abs``, ``neg``, ``floor``, ``ceil``, ``round``, ``sqrt``,
``rsqrt``, ``exp``, ``log``, ``sin``, ``cos``, ``tan``, ``asin``, ``acos``,
``atan``, ``sinh``, ``cosh``, ``sign``, ``not_``.

All binary ops handle scalar arguments and broadcasting via
``get_trt_tensor``/``broadcastable_fn`` utilities.

embedding
^^^^^^^^^

**``impl.embedding``** — Implements ``torch.nn.Embedding`` and
``torch.nn.EmbeddingBag`` via ``add_gather`` (index gather on the weight matrix).

full
^^^^

**``impl.full``** — Creates constant fill tensors (``torch.full``, ``torch.zeros``,
``torch.ones``) via ``IFillLayer``.

grid
^^^^

**``impl.grid``** — Grid sampling (``torch.nn.functional.grid_sample``) via
TRT's ``IGridSampleLayer``.

linear
^^^^^^

**``impl.linear``** — Fully-connected layer (``torch.nn.Linear``).
Wraps ``add_matrix_multiply`` + optional ``add_elementwise`` for bias.

matmul
^^^^^^

**``impl.matmul``** — General matrix multiplication. Handles batched matmul,
dot products, and outer products by reshaping inputs before calling
``add_matrix_multiply``.

nccl_ops
^^^^^^^^^

**``impl.nccl_ops``** — Fused NCCL collective wrappers for distributed inference
(``all_gather``, ``reduce_scatter``). Used by the ``fuse_distributed_ops`` lowering
pass.

normalization
^^^^^^^^^^^^^

**``impl.normalization``** — Normalization layers: ``batch_norm``,
``layer_norm``, ``group_norm``, ``instance_norm``, ``rms_norm``.

``batch_norm`` handles constant-folded running stats (for inference) and
delegates to ``add_scale`` for the affine transform.

pad
^^^

**``impl.pad``** — Tensor padding: ``constant_pad``, ``reflection_pad``,
``replication_pad``. Wraps ``add_padding_nd`` or ``add_slice`` depending on
padding mode.

permutation
^^^^^^^^^^^

**``impl.permutation``** — ``transpose``, ``permute``. Wraps ``add_shuffle``
with a permuted reshape.

pool
^^^^

**``impl.pool``** — Pooling: ``avg_pool``, ``max_pool``,
``adaptive_avg_pool``, ``adaptive_max_pool``. Handles 1D/2D/3D and the
``ceil_mode`` / ``count_include_pad`` options via ``add_pooling_nd``.

prelu
^^^^^

**``impl.prelu``** — Parametric ReLU. Implemented as
``max(0, x) + slope * min(0, x)`` using elementwise ops.

quantize
^^^^^^^^

**``impl.quantize``** — Quantize / dequantize layers for INT8 and FP8
calibration workflows. Wraps ``add_quantize`` / ``add_dequantize``.

reduce
^^^^^^

**``impl.reduce``** — Reduction operations: ``sum``, ``mean``, ``max``,
``min``, ``prod``, ``any``, ``all``, ``norm``, ``var``, ``std``.
Wraps ``add_reduce`` with appropriate ``trt.ReduceOperation``.

select
^^^^^^

**``impl.select``** — Index selection and slicing primitives: ``gather``,
``index``, ``index_select``, ``gather_nd``. Wraps ``add_gather`` variants.

shape
^^^^^

**``impl.shape``** — Shape introspection: ``size``, ``numel``, ``shape``.
Returns ``ITensor`` objects holding shape values for use in dynamic-shape graphs.

shuffle
^^^^^^^

**``impl.shuffle``** — Reshape and view via ``add_shuffle``. Used by converters
for ``view``, ``reshape``, ``flatten``, ``unsqueeze``, ``squeeze``.

slice
^^^^^

**``impl.slice``** (available but not listed in ``__init__.py`` directly)
— Slicing and strided access via ``add_slice``. Used by converters for
``aten.slice.Tensor`` and ``aten.select.int``.

split
^^^^^

**``impl.split``** — Splits a tensor into chunks along a dimension.
Implemented via repeated ``add_slice``.

squeeze / unsqueeze
^^^^^^^^^^^^^^^^^^^^

**``impl.squeeze``** / **``impl.unsqueeze``** — Remove / add size-1 dimensions.
Both delegate to ``impl.shuffle`` with a reshaped output spec.

topk
^^^^

**``impl.topk``** — ``torch.topk``. Wraps ``add_topk`` with ascending/descending
support.

upsample
^^^^^^^^

**``impl.upsample``** — ``torch.nn.functional.interpolate`` (nearest and bilinear).
Wraps ``add_resize`` with the appropriate resize mode.

----

SourceIR
----------

.. _source_ir:

Every ``impl`` function takes a ``source_ir: Optional[SourceIR]`` argument that tags
the origin of the call for debug layer naming:

.. code-block:: python

    class SourceIR(Enum):
        ATEN    = auto()   # Called from an ATen converter
        TORCHSCRIPT = auto()  # Called from a TorchScript converter (legacy)
        NN      = auto()   # Called from an nn.Module-level converter
        ACC     = auto()   # Called from an ACC (operator fusion) converter
        PRIM    = auto()   # Called from a prims-level converter
        CORE_ATEN = auto() # Called from a Core ATen op converter
        UNKNOWN = auto()

The value is appended to the generated TRT layer name, making engine profiling and
debugging easier. Always pass the appropriate value — use ``SourceIR.ATEN`` for all
standard ATen-based converters.
