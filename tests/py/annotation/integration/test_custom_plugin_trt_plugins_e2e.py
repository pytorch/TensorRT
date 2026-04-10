"""End-to-end tests for trt_plugins.custom_op(impl=tta.custom_plugin(...)).

Each test compiles the model with torch_tensorrt and verifies:
  1. The custom op lowers into exactly one TRT engine.
  2. That engine contains a PluginV3 layer for the custom op.
  3. The engine output matches a pure-PyTorch reference.

Backends and operations
-----------------------
  Backend   | Unary            | Binary
  ----------|------------------|----------------------------------
  Triton    | silu  (x·σ(x))  | swiglu  (silu(gate)·up)
  CuTile    | relu  (max(x,0)) | reglu   (relu(gate)·up)
  CuTeDSL   | silu  (x·σ(x))  | hadamard (x·y)

Triton and CuTile also register multi-config variants (BLOCK_SIZE ∈
{64, 128, 256}) so TRT's tactic-selection autotuner is exercised.

Test semantics (shared across all backends via _BackendE2ETests mixin)
----------------------------------------------------------------------
  test_unary_activation        : standalone activation on [seq, hidden]
  test_binary_gating           : standalone gating on [seq, hidden] pairs
  test_llm_hidden_unary        : activation at LLM hidden dim [batch=8, hidden=512]
  test_llm_hidden_binary       : gating at LLM hidden dim [batch=8, hidden=512]
  test_dynamic_batch           : dynamic batch dimension, hidden=256
  test_gated_ffn_llm           : LLM-ratio FFN (hidden=256, inter=512, 2× expansion)
  test_gated_ffn_block         : shared-input FFN (xfail — TRT mergeMatmulLayers bug)
  test_gated_ffn_block_contiguous : separate-input FFN workaround
  test_chained_silu_gate       : two chained PluginV3 ops in one engine

Multi-config semantics (Triton + CuTile only, via _MultiConfigTests mixin)
--------------------------------------------------------------------------
  test_multi_config_unary   : unary op with 3 tile-size configs, TRT picks best
  test_multi_config_binary  : binary op with 3 tile-size configs, TRT picks best

Cross-backend semantics (TestCrossBackendE2E)
---------------------------------------------
  Tests that mix plugins from different backends in a single TRT engine,
  verifying that multiple QDP PluginV3 layers coexist and compile correctly.
  All shapes are LLM-domain: [batch=8, hidden=512].
"""

import math
import unittest

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
import torch_tensorrt.annotation as tta
import torch_tensorrt.dynamo.conversion.plugins as trt_plugins
import triton
import triton.language as tl
import cuda.tile as ct
import cutlass.cute as cute
from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import PythonTorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

def _get_trt_engines(model):
    """Return all TRT engine submodules in a compiled model."""
    return [
        m for m in model.modules()
        if isinstance(m, (PythonTorchTensorRTModule, TorchTensorRTModule))
    ]


def _engine_has_pluginv3_layer(engine, op_name):
    """Return True iff the engine contains a PluginV3 layer for *op_name*.

    *op_name* is 'namespace::name' (e.g. 'torchtrt_e2e_triton::swiglu').

    Two conditions:
      1. The op's creator in the TRT registry is _TemplatePluginCreator (QDP /
         PluginV3), not the legacy IPluginCreator (V2).
      2. A layer whose name contains '/<name>' appears in the engine.
         Two naming conventions are supported:
         - Direct: the layer name starts with '/<name>' (e.g. when set via
           lower_to_trt which uses ``plugin_layer.name = name``).
         - Wrapped: the layer name contains '-[/<name>' (e.g. when set via
           generate_plugin_converter which uses
           ``layer.name = f"[{target}]-[{name}]"``).
         TRT's Myelin optimizer may also append suffixes such as _myl0_0, so
         substring matching is used rather than exact string equality.
    """
    namespace, name = op_name.split("::", 1)

    reg = trt.get_plugin_registry()
    creator = reg.get_creator(name, "1", namespace)
    if creator is None or type(creator).__name__ == "IPluginCreator":
        return False

    insp = engine.create_engine_inspector()
    layer_names = [
        insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE).strip()
        for i in range(engine.num_layers)
    ]
    prefix = f"/{name}"
    return any(
        ln == prefix or ln.startswith(prefix + "_") or f"-[{prefix}" in ln
        for ln in layer_names
    )


def _engine_all_layer_names(engine) -> list:
    """Return all layer name strings from the TRT engine inspector."""
    insp = engine.create_engine_inspector()
    return [
        insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE).strip()
        for i in range(engine.num_layers)
    ]


def _trt_compile(model, inputs):
    torch._dynamo.reset()
    return torch_tensorrt.compile(model, inputs=inputs, min_block_size=1)


def _check(test, model, inputs, ref_fn, op_name, rtol=1e-3, atol=1e-3):
    """Compile model to TRT and assert output matches *ref_fn(*inputs)*.

    Also asserts:
      - exactly one TRT engine segment is produced
      - that engine contains a PluginV3 layer for *op_name*

    *rtol* / *atol* are forwarded to assert_close.  Tests involving chained
    linear layers may pass looser tolerances due to FP32 accumulation across
    multiple matrix multiplications.
    """
    ref = ref_fn(*inputs)
    compiled = _trt_compile(model, inputs)
    engines = _get_trt_engines(compiled)
    test.assertEqual(len(engines), 1,
                     "Expected the custom op to lower into exactly one TRT engine")
    test.assertTrue(
        _engine_has_pluginv3_layer(engines[0].engine, op_name),
        f"Expected a PluginV3 layer for '{op_name}' in the TRT engine",
    )
    trt_out = compiled(*inputs)
    torch.testing.assert_close(trt_out, ref, rtol=rtol, atol=atol)


def _check_multi_op(test, model, inputs, ref_fn, op_names, rtol=1e-3, atol=1e-3):
    """Like _check but verifies multiple PluginV3 layers coexist in one engine."""
    ref = ref_fn(*inputs)
    compiled = _trt_compile(model, inputs)
    engines = _get_trt_engines(compiled)
    test.assertEqual(len(engines), 1,
                     "Expected all custom ops to lower into exactly one TRT engine")
    for op_name in op_names:
        test.assertTrue(
            _engine_has_pluginv3_layer(engines[0].engine, op_name),
            f"Expected a PluginV3 layer for '{op_name}' in the TRT engine",
        )
    trt_out = compiled(*inputs)
    torch.testing.assert_close(trt_out, ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Reusable nn.Module wrappers
# ---------------------------------------------------------------------------

class _UnaryOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self._op = op

    def forward(self, x):
        return self._op(x)


class _BinaryOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self._op = op

    def forward(self, x, y):
        return self._op(x, y)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementations
# ---------------------------------------------------------------------------

def _ref_silu(x):
    """SiLU / Swish: x · σ(x).  Used in LLaMA, Mistral, Phi, etc."""
    return x * torch.sigmoid(x)


def _ref_swiglu(gate, up):
    """SwiGLU gate: silu(gate) · up.  Used in LLaMA / Mistral FFN blocks."""
    return gate * torch.sigmoid(gate) * up


def _ref_relu(x):
    return torch.relu(x)


def _ref_reglu(gate, up):
    """ReGLU gate: relu(gate) · up.  A simpler gated activation."""
    return torch.relu(gate) * up


def _ref_hadamard(x, y):
    """Element-wise product.  Used in attention masking, LoRA updates, etc."""
    return x * y


# ---------------------------------------------------------------------------
# Triton kernels
#
# Implements SiLU (x·σ(x)) and SwiGLU (silu(gate)·up).
# These are the activation and gating ops used verbatim in the LLaMA-series
# feed-forward blocks.  tl.sigmoid is a first-class Triton intrinsic.
# ---------------------------------------------------------------------------

@triton.jit
def _triton_silu_kernel(x_ptr, out_ptr, n_cols,
                        x_stride0, x_stride1, out_stride0, out_stride1,
                        BLOCK_SIZE: tl.constexpr):
    """Stride-aware SiLU: 2D grid, fully general per-dim strides.

    grid = (n_rows, cdiv(n_cols, BLOCK_SIZE)).
    program_id(0) = row; program_id(1) = column-block.
    Pointer offset: row * stride0 + col * stride1.
    Both stride dimensions are passed so the kernel handles any tensor layout
    (contiguous, row-padded LINEAR, or any stride(dim) value).
    """
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_stride0 + col_offsets * x_stride1, mask=mask)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1,
             x * tl.sigmoid(x), mask=mask)


def _triton_launch_silu(x, out, BLOCK_SIZE=128):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_silu_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, out, n_cols,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _triton_swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_cols,
                           gate_stride0, gate_stride1,
                           up_stride0, up_stride1,
                           out_stride0, out_stride1,
                           BLOCK_SIZE: tl.constexpr):
    """Stride-aware SwiGLU: 2D grid, fully general per-dim strides.

    grid = (n_rows, cdiv(n_cols, BLOCK_SIZE)).
    program_id(0) = row; program_id(1) = column-block.
    Pointer offset: row * stride0 + col * stride1 for each tensor.
    Both stride dimensions are passed so the kernel handles any tensor layout
    (contiguous, row-padded LINEAR, or any stride(dim) value).
    """
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    gate = tl.load(gate_ptr + row * gate_stride0 + col_offsets * gate_stride1, mask=mask)
    up = tl.load(up_ptr + row * up_stride0 + col_offsets * up_stride1, mask=mask)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1,
             gate * tl.sigmoid(gate) * up, mask=mask)


def _triton_launch_swiglu(gate, up, out, BLOCK_SIZE=128):
    n_rows, n_cols = gate.shape[0], gate.shape[1]
    _triton_swiglu_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        gate, up, out, n_cols,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_triton::swiglu",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_swiglu, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda gate, up: gate.new_empty(gate.shape),
    ),
    supports_dynamic_shapes=True,
)

# Multi-config: TRT benchmarks BLOCK_SIZE ∈ {64, 128, 256} and picks the
# fastest tactic for the target GPU automatically at engine-build time.
trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_mc",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu,
                   configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_triton::swiglu_mc",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_swiglu,
                   configs=[{"BLOCK_SIZE": 64}, {"BLOCK_SIZE": 128}, {"BLOCK_SIZE": 256}]),
        meta_impl=lambda gate, up: gate.new_empty(gate.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# CuTile kernels
#
# Implements ReLU (max(x, 0)) and ReGLU (relu(gate)·up).
# ReGLU is a simpler gated variant of the SwiGLU pattern; it appears in
# ablation studies of transformer architectures and in quantized deployments
# where ReLU is preferred for sparsity.
# ---------------------------------------------------------------------------

@ct.kernel
def _ct_relu_kernel(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    ct.store(out, index=(pid,), tile=ct.maximum(x_tile, 0.0))


def _ct_launch_relu(x, out, BLOCK=128):
    n = x.numel()
    stream = torch.cuda.current_stream().cuda_stream
    # CuTile kernels use a 1D tile index. In the AOT sandbox path tensors are
    # SymbolicTensor; skip reshape/contiguous there — the sandbox only needs
    # the grid tuple and never executes the kernel body.
    if isinstance(x, torch.Tensor):
        x = x.contiguous().reshape(-1)
        out = out.reshape(-1)
    ct.launch(stream, (ct.cdiv(n, BLOCK), 1, 1), _ct_relu_kernel, (x, out, BLOCK))


@ct.kernel
def _ct_reglu_kernel(gate, up, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    g_tile = ct.load(gate, index=(pid,), shape=(tile_size,))
    u_tile = ct.load(up, index=(pid,), shape=(tile_size,))
    ct.store(out, index=(pid,), tile=ct.maximum(g_tile, 0.0) * u_tile)


def _ct_launch_reglu(gate, up, out, BLOCK=128):
    n = gate.numel()
    stream = torch.cuda.current_stream().cuda_stream
    if isinstance(gate, torch.Tensor):
        gate = gate.contiguous().reshape(-1)
        up = up.contiguous().reshape(-1)
        out = out.reshape(-1)
    ct.launch(stream, (ct.cdiv(n, BLOCK), 1, 1), _ct_reglu_kernel, (gate, up, out, BLOCK))


trt_plugins.custom_op(
    "torchtrt_e2e_cutile::relu",
    impl=tta.custom_plugin(
        tta.cutile(_ct_launch_relu, configs=[{"BLOCK": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_cutile::reglu",
    impl=tta.custom_plugin(
        tta.cutile(_ct_launch_reglu, configs=[{"BLOCK": 128}]),
        meta_impl=lambda gate, up: gate.new_empty(gate.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_cutile::relu_mc",
    impl=tta.custom_plugin(
        tta.cutile(_ct_launch_relu,
                   configs=[{"BLOCK": 64}, {"BLOCK": 128}, {"BLOCK": 256}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_cutile::reglu_mc",
    impl=tta.custom_plugin(
        tta.cutile(_ct_launch_reglu,
                   configs=[{"BLOCK": 64}, {"BLOCK": 128}, {"BLOCK": 256}]),
        meta_impl=lambda gate, up: gate.new_empty(gate.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# CuTeDSL kernels
#
# Implements SiLU (x·σ(x)) and Hadamard product (x·y).
# Sigmoid is computed as 1 / (1 + exp(-x)) via cute.arch.exp, which
# lowers to the CUDA expf intrinsic in device code.
# ---------------------------------------------------------------------------

@cute.kernel
def _cute_silu_kernel(x: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    xi = x[idx]
    one = x.element_type(1.0)
    sigmoid_xi = one / (one + cute.arch.exp(x.element_type(-1.0) * xi))
    out[idx] = xi * sigmoid_xi


@cute.jit
def _cute_launch_silu(x: cute.Tensor, out: cute.Tensor):
    _cute_silu_kernel(x, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


@cute.kernel
def _cute_hadamard_kernel(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    idx = cute.arch.block_idx()[0]
    out[idx] = x[idx] * y[idx]


@cute.jit
def _cute_launch_hadamard(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    _cute_hadamard_kernel(x, y, out).launch(grid=(math.prod(x.shape), 1, 1), block=(1, 1, 1))


trt_plugins.custom_op(
    "torchtrt_e2e_cutedsl::silu",
    impl=tta.custom_plugin(
        tta.cutedsl(_cute_launch_silu),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)

trt_plugins.custom_op(
    "torchtrt_e2e_cutedsl::hadamard",
    impl=tta.custom_plugin(
        tta.cutedsl(_cute_launch_hadamard),
        meta_impl=lambda x, y: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Plugin attrs: Triton SiLU with a compile-time SCALE factor passed via attrs
#
# SCALE is declared as tl.constexpr so Triton bakes it into the PTX.
# Passing SCALE=2.0 directly as a kwarg to tta.custom_plugin() stores it in
# CustomPluginSpec.attrs; the aot_impl path merges attrs into the sandbox
# kwargs so the kernel is compiled with the correct constant.
# ---------------------------------------------------------------------------

@triton.jit
def _triton_scaled_silu_kernel(x_ptr, out_ptr, n_cols,
                                x_stride0, x_stride1, out_stride0, out_stride1,
                                SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """SiLU scaled by a compile-time constant: out = SCALE * x * σ(x)."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_stride0 + col_offsets * x_stride1, mask=mask)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1,
             x * tl.sigmoid(x) * SCALE, mask=mask)


def _triton_launch_scaled_silu(x, out, BLOCK_SIZE=128, SCALE=1.0):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_scaled_silu_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, out, n_cols,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        SCALE=SCALE, BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_scaled",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_scaled_silu, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
        SCALE=2.0,  # direct kwarg: baked into PTX as tl.constexpr
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Multiple outputs: Triton kernel that returns two tensors from one op.
#
# The op computes both SiLU and ReLU activations in a single fused kernel,
# returning them as a (silu_out, relu_out) pair.  This exercises the
# multi-output QDP path:
#   meta_impl returning tuple → _infer_num_outputs=2
#   → @trtp.register with Tuple[TensorDesc, TensorDesc] return type
#   → TRT PluginV3 layer with num_outputs=2
# ---------------------------------------------------------------------------

@triton.jit
def _triton_dual_kernel(x_ptr, silu_ptr, relu_ptr, n_cols,
                         x_s0, x_s1, silu_s0, silu_s1, relu_s0, relu_s1,
                         BLOCK_SIZE: tl.constexpr):
    """Fused SiLU + ReLU: silu_out = x·σ(x), relu_out = max(x, 0)."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_s0 + col_offsets * x_s1, mask=mask)
    tl.store(silu_ptr + row * silu_s0 + col_offsets * silu_s1,
             x * tl.sigmoid(x), mask=mask)
    tl.store(relu_ptr + row * relu_s0 + col_offsets * relu_s1,
             tl.maximum(x, 0.0), mask=mask)


def _triton_launch_dual(x, silu_out, relu_out, BLOCK_SIZE=128):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_dual_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, silu_out, relu_out, n_cols,
        x.stride(0), x.stride(1),
        silu_out.stride(0), silu_out.stride(1),
        relu_out.stride(0), relu_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::dual_activation",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_dual, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: (x.new_empty(x.shape), x.new_empty(x.shape)),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# supports_dynamic_shapes=False: static-only engine
#
# This op is identical to silu but registered with supports_dynamic_shapes=False.
# ---------------------------------------------------------------------------

trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_static",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=False,
)


# ---------------------------------------------------------------------------
# Tensor format: explicit input_formats / output_formats on a Triton spec
# ---------------------------------------------------------------------------

trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_linear_fmt",
    impl=tta.custom_plugin(
        tta.triton(
            _triton_launch_silu,
            configs=[{"BLOCK_SIZE": 128}],
            input_formats=[trt.TensorFormat.LINEAR],
            output_formats=[trt.TensorFormat.LINEAR],
        ),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Non-LINEAR format: CHW32 Triton plugin
#
# A flat-indexed (layout-agnostic) elementwise ReLU registered with
# input_formats=[trt.TensorFormat.CHW32].  TRT will negotiate CHW32 format
# and insert reformatting nodes (LINEAR→CHW32 before, CHW32→LINEAR after)
# around the plugin layer.
#
# The kernel uses flat pointer + offset arithmetic so it is correct for any
# contiguous memory layout, including CHW32.  A [4, 32, 8, 8] input uses
# 32 channels to satisfy CHW32's channel-alignment requirement.
# ---------------------------------------------------------------------------


@triton.jit
def _triton_flat_relu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    """Elementwise ReLU via flat pointer offset — layout-agnostic."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def _triton_launch_flat_relu(x, out, BLOCK=128):
    n = x.numel()
    _triton_flat_relu_kernel[(math.ceil(n / BLOCK),)](x, out, n=n, BLOCK=BLOCK)


trt_plugins.custom_op(
    "torchtrt_e2e_triton::relu_chw32",
    impl=tta.custom_plugin(
        tta.triton(
            _triton_launch_flat_relu,
            configs=[{"BLOCK": 128}],
            input_formats=[trt.TensorFormat.CHW32],
            output_formats=[trt.TensorFormat.CHW32],
        ),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=False,
)


# ---------------------------------------------------------------------------
# Float16 dtype: Triton SiLU with explicit fp32 promotion for AOT compatibility
#
# Triton's AOT compiler can fail for tl.sigmoid with *fp16 pointer types in
# some builds.  This kernel promotes fp16→fp32 for sigmoid, then casts back.
# ---------------------------------------------------------------------------

@triton.jit
def _triton_silu_f16_kernel(x_ptr, out_ptr, n_cols,
                              x_stride0, x_stride1, out_stride0, out_stride1,
                              BLOCK_SIZE: tl.constexpr):
    """FP16 SiLU: load fp16, promote to fp32 for sigmoid, store fp16."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_stride0 + col_offsets * x_stride1, mask=mask)
    xf = x.to(tl.float32)
    result = (xf * tl.sigmoid(xf)).to(tl.float16)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1, result, mask=mask)


def _triton_launch_silu_f16(x, out, BLOCK_SIZE=128):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_silu_f16_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, out, n_cols,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_f16",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu_f16, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Weights: Triton SiLU with a constant column-scale weight tensor.
#
# W is a [hidden_dim] weight baked into the TRT engine as a trt.add_constant
# layer.  The plugin receives (x, W) as inputs; out[i,j] = W[j] * silu(x[i,j]).
# This exercises the full custom_plugin(W=tensor) → weights injection path.
#
# _W_COLUMN_SCALE is a module-level constant so its values are known and the
# test reference can be computed independently without calling the torch op.
# ---------------------------------------------------------------------------

_LLM_B, _LLM_H = 8, 512  # LLM-domain shape used for cross-backend and weight-injection tests

_W_COLUMN_SCALE = torch.full((_LLM_H,), 3.0)  # CPU tensor; lowered via trt.add_constant


@triton.jit
def _triton_weighted_silu_kernel(x_ptr, w_ptr, out_ptr, n_cols,
                                   x_s0, x_s1, w_s0, out_s0, out_s1,
                                   BLOCK_SIZE: tl.constexpr):
    """Elementwise W[j] * silu(x[i,j]) for a fixed column-scale weight W."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_s0 + col_offsets * x_s1, mask=mask)
    w = tl.load(w_ptr + col_offsets * w_s0, mask=mask)
    tl.store(out_ptr + row * out_s0 + col_offsets * out_s1,
             x * tl.sigmoid(x) * w, mask=mask)


def _triton_launch_weighted_silu(x, W, out, BLOCK_SIZE=128):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_weighted_silu_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, W, out, n_cols,
        x.stride(0), x.stride(1), W.stride(0), out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::weighted_silu",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_weighted_silu, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
        W=_W_COLUMN_SCALE,  # tensor kwarg → injected as trt.add_constant
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Flat (rank-agnostic) Triton SiLU: handles any-rank input via numel().
#
# The 2D silu kernel assumes rank-2 input.  This flat variant uses a single
# 1D grid over all elements, making it compatible with 3D inputs [B, S, H].
# The launch function reshapes real tensors to 1D before calling the kernel;
# for SymbolicTensors (sandbox path) reshape is skipped since the kernel
# recording only needs the grid shape and pointer types.
# ---------------------------------------------------------------------------

@triton.jit
def _triton_silu_flat_kernel(x_ptr, out_ptr, n_total, BLOCK_SIZE: tl.constexpr):
    """Flat SiLU over n_total elements; grid = (cdiv(n_total, BLOCK_SIZE),)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_total
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * tl.sigmoid(x), mask=mask)


def _triton_launch_silu_flat(x, out, BLOCK_SIZE=128):
    n = x.numel()
    if isinstance(x, torch.Tensor):
        x = x.contiguous().reshape(-1)
        out = out.reshape(-1)
    _triton_silu_flat_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        x, out, n, BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_flat",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu_flat, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# BFloat16 dtype: Triton SiLU with explicit fp32 promotion.
#
# Mirrors the fp16 variant: loads bf16, promotes to fp32 for sigmoid, stores bf16.
# ---------------------------------------------------------------------------

@triton.jit
def _triton_silu_bf16_kernel(x_ptr, out_ptr, n_cols,
                               x_stride0, x_stride1, out_stride0, out_stride1,
                               BLOCK_SIZE: tl.constexpr):
    """BF16 SiLU: load bf16, promote to fp32 for sigmoid, store bf16."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_stride0 + col_offsets * x_stride1, mask=mask)
    xf = x.to(tl.float32)
    result = (xf * tl.sigmoid(xf)).to(tl.bfloat16)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1, result, mask=mask)


def _triton_launch_silu_bf16(x, out, BLOCK_SIZE=128):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_silu_bf16_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, out, n_cols,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_bf16",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_silu_bf16, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Multiple attrs: Triton SiLU with two compile-time constants SCALE and BIAS.
#
# Both are declared tl.constexpr and baked into the PTX at AOT time.
# This exercises the full multi-attr path:
#   custom_plugin(SCALE=3.0, BIAS=1.0) → attrs={"SCALE":3.0,"BIAS":1.0}
#   → sandbox merged kwargs → constexpr values in PTX
# ---------------------------------------------------------------------------

@triton.jit
def _triton_scale_bias_kernel(x_ptr, out_ptr, n_cols,
                                x_stride0, x_stride1, out_stride0, out_stride1,
                                SCALE: tl.constexpr, BIAS: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr):
    """SiLU scaled and shifted: out = SCALE * x * σ(x) + BIAS."""
    row = tl.program_id(0)
    col_pid = tl.program_id(1)
    col_offsets = col_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(x_ptr + row * x_stride0 + col_offsets * x_stride1, mask=mask)
    tl.store(out_ptr + row * out_stride0 + col_offsets * out_stride1,
             x * tl.sigmoid(x) * SCALE + BIAS, mask=mask)


def _triton_launch_scale_bias(x, out, BLOCK_SIZE=128, SCALE=1.0, BIAS=0.0):
    n_rows, n_cols = x.shape[0], x.shape[1]
    _triton_scale_bias_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
        x, out, n_cols,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        SCALE=SCALE, BIAS=BIAS, BLOCK_SIZE=BLOCK_SIZE,
    )


trt_plugins.custom_op(
    "torchtrt_e2e_triton::silu_scale_bias",
    impl=tta.custom_plugin(
        tta.triton(_triton_launch_scale_bias, configs=[{"BLOCK_SIZE": 128}]),
        meta_impl=lambda x: x.new_empty(x.shape),
        SCALE=3.0, BIAS=1.0,
    ),
    supports_dynamic_shapes=True,
)


# ---------------------------------------------------------------------------
# Mixin: shared test semantics across all backends
# ---------------------------------------------------------------------------

class _BackendE2ETests:
    """Four core E2E semantics shared by every backend.

    Concrete subclasses must set:
      _NS          : op namespace (e.g. "torchtrt_e2e_triton")
      _UNARY_OP    : unary op name (e.g. "silu")
      _BINARY_OP   : binary op name (e.g. "swiglu")
      _UNARY_REF   : staticmethod — pure-PyTorch reference for the unary op
      _BINARY_REF  : staticmethod — pure-PyTorch reference for the binary op
    """

    _NS: str
    _UNARY_OP: str
    _BINARY_OP: str
    _UNARY_REF: callable
    _BINARY_REF: callable

    def _op(self, name):
        return getattr(getattr(torch.ops, self._NS), name).default

    def test_unary_activation(self):
        """Standalone activation on [seq_len=16, hidden=256] — typical LLM hidden state.

        Exercises the op in isolation: no surrounding aten ops, single custom
        plugin layer in the engine.
        """
        m = _UnaryOp(self._op(self._UNARY_OP)).eval().cuda()
        inputs = [torch.randn(16, 256, device="cuda")]
        _check(self, m, inputs, self._UNARY_REF, f"{self._NS}::{self._UNARY_OP}")

    def test_binary_gating(self):
        """Standalone gating on [seq_len=16, hidden=256] tensor pairs.

        Exercises the binary custom op in isolation with matched input shapes,
        reproducing the element-wise gate × value combination used in attention
        and gated linear units before the down-projection.
        """
        m = _BinaryOp(self._op(self._BINARY_OP)).eval().cuda()
        inputs = [torch.randn(16, 256, device="cuda"), torch.randn(16, 256, device="cuda")]
        _check(self, m, inputs, self._BINARY_REF, f"{self._NS}::{self._BINARY_OP}")

    # TRT mergeMatmulLayers delivers non-contiguous sub-region buffers to
    # IPluginV3::enqueue without inserting a reformat copy, violating the
    # LINEAR stride contract.  Expected to fail until TRT fixes this.
    @unittest.expectedFailure
    def test_gated_ffn_block(self):
        """Full gated FFN block with shared input — expected to fail due to TRT bug.

        This is the exact feed-forward structure used in LLaMA, Mistral, Qwen,
        and other SwiGLU / ReGLU-based transformers:

            gate = fc_gate(x)           # [batch, hidden] → [batch, intermediate]
            up   = fc_up(x)             # [batch, hidden] → [batch, intermediate]
            h    = gate_fn(gate, up)    # custom PluginV3 — fused gate computation
            out  = fc_down(h)           # [batch, intermediate] → [batch, hidden]

        Both fc_gate and fc_up share the same input ITensor x. TRT's
        mergeMatmulLayers optimizer fuses them into a single [batch, 2*intermediate]
        GEMM and presents each half as a [batch, intermediate] sub-region with
        physical row-stride 2*intermediate while still tagging the format as LINEAR.
        This violates the LINEAR stride contract: IPluginV3::enqueue receives
        non-contiguous buffers but PluginTensorDesc has no physical stride field,
        so the plugin infers stride = intermediate (logical) instead of 2*intermediate.

        See test_gated_ffn_block_contiguous for the workaround (separate inputs).
        TRT bug filed: mergeMatmulLayers delivers non-contiguous sub-region buffers
        to IPluginV3::enqueue without inserting a reformat copy.
        """
        hidden_dim, intermediate_dim = 64, 128
        gate_op = self._op(self._BINARY_OP)
        _binary_ref = self._BINARY_REF

        class GatedFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
                self._gate = gate_op

            def forward(self, x):
                gate = self.fc_gate(x)
                up = self.fc_up(x)
                return self.fc_down(self._gate(gate, up))

        m = GatedFFN().eval().cuda()
        inputs = [torch.randn(8, hidden_dim, device="cuda")]

        def _ref(x):
            with torch.no_grad():
                return m.fc_down(_binary_ref(m.fc_gate(x), m.fc_up(x)))

        _check(self, m, inputs, _ref, f"{self._NS}::{self._BINARY_OP}", rtol=1e-3, atol=1e-3)

    def test_gated_ffn_block_contiguous(self):
        """Full gated FFN block with separate inputs — workaround for TRT mergeMatmulLayers bug.

        Same four-op structure as test_gated_ffn_block but fc_gate and fc_up
        receive separate input ITensors (x and x_up), which prevents TRT's
        mergeMatmulLayers from fusing the two matmuls.  Each matmul produces its
        own contiguous [batch, intermediate] buffer; the plugin receives
        contiguous inputs and the stride-aware kernel reads correct data.

        The model forward takes (x, x_up) where the caller passes the same
        tensor for both; in the TRT graph they are distinct network inputs
        so canMatmulBeHorizontallyMerged returns false.
        """
        hidden_dim, intermediate_dim = 64, 128
        gate_op = self._op(self._BINARY_OP)
        _binary_ref = self._BINARY_REF

        class GatedFFNContiguous(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
                self._gate = gate_op

            def forward(self, x, x_up):
                gate = self.fc_gate(x)
                up = self.fc_up(x_up)
                return self.fc_down(self._gate(gate, up))

        m = GatedFFNContiguous().eval().cuda()
        x = torch.randn(8, hidden_dim, device="cuda")
        inputs = [x, x]

        def _ref(x, x_up):
            with torch.no_grad():
                return m.fc_down(_binary_ref(m.fc_gate(x), m.fc_up(x_up)))

        _check(self, m, inputs, _ref, f"{self._NS}::{self._BINARY_OP}", rtol=1e-3, atol=1e-3)

    def test_llm_hidden_unary(self):
        """Standalone activation at LLM hidden dimension [batch=8, hidden=512].

        Tests the activation at the scale of a small LLM hidden state, exercising
        tactic selection at tensor sizes representative of production inference.
        """
        m = _UnaryOp(self._op(self._UNARY_OP)).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        _check(self, m, inputs, self._UNARY_REF, f"{self._NS}::{self._UNARY_OP}")

    def test_llm_hidden_binary(self):
        """Standalone gating at LLM hidden dimension [batch=8, hidden=512].

        Tests the binary gating op at the scale of a small LLM hidden state,
        covering the element-wise gate × value step in real SwiGLU / ReGLU FFNs.
        """
        m = _BinaryOp(self._op(self._BINARY_OP)).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda"), torch.randn(8, 512, device="cuda")]
        _check(self, m, inputs, self._BINARY_REF, f"{self._NS}::{self._BINARY_OP}")

    def test_dynamic_batch(self):
        """Unary op with dynamic batch dimension, hidden=256.

        Uses torch_tensorrt.Input with min/opt/max batch sizes to exercise TRT's
        dynamic shape path.  Compilation uses opt_shape=8; inference runs at batch=8.
        """
        hidden = 256
        m = _UnaryOp(self._op(self._UNARY_OP)).eval().cuda()
        torch._dynamo.reset()
        compiled = torch_tensorrt.compile(
            m,
            inputs=[torch_tensorrt.Input(
                min_shape=(1, hidden), opt_shape=(8, hidden), max_shape=(32, hidden),
                dtype=torch.float32,
            )],
            min_block_size=1,
        )
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1,
                         "Expected the custom op to lower into exactly one TRT engine")
        self.assertTrue(
            _engine_has_pluginv3_layer(engines[0].engine, f"{self._NS}::{self._UNARY_OP}"),
            f"Expected a PluginV3 layer for '{self._NS}::{self._UNARY_OP}' in the TRT engine",
        )
        x = torch.randn(8, hidden, device="cuda")
        trt_out = compiled(x)
        torch.testing.assert_close(trt_out, self._UNARY_REF(x), rtol=1e-3, atol=1e-3)

    def test_gated_ffn_llm(self):
        """Full gated FFN at LLM expansion ratio: hidden=256, inter=512 (2×).

        Same four-op structure as test_gated_ffn_block_contiguous but at LLM
        scale: hidden=256, intermediate=512 reproduces the 2× expansion ratio
        used in LLaMA / Mistral FFN blocks.  Two distinct inputs prevent
        mergeMatmulLayers fusion; both matmuls produce contiguous buffers.
        """
        hidden_dim, intermediate_dim = 256, 512
        gate_op = self._op(self._BINARY_OP)
        _binary_ref = self._BINARY_REF

        class GatedFFNLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
                self.fc_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
                self._gate = gate_op

            def forward(self, x, x_up):
                gate = self.fc_gate(x)
                up = self.fc_up(x_up)
                return self.fc_down(self._gate(gate, up))

        m = GatedFFNLLM().eval().cuda()
        x = torch.randn(8, hidden_dim, device="cuda")
        inputs = [x, x]

        def _ref(x, x_up):
            with torch.no_grad():
                return m.fc_down(_binary_ref(m.fc_gate(x), m.fc_up(x_up)))

        _check(self, m, inputs, _ref, f"{self._NS}::{self._BINARY_OP}", rtol=1e-3, atol=1e-3)

    def test_chained_silu_gate(self):
        """Chained custom ops reproducing the core of SwiGLU / ReGLU gating.

        The two-op chain:
            activated = unary_act(x)          ← first PluginV3
            out       = binary_gate(activated, y)  ← second PluginV3

        models the split where x is the pre-activated gate branch and y is
        the up branch that has been computed separately (e.g. by a separate
        linear projection not shown here).  Both ops must land in one engine.
        """
        act_op = self._op(self._UNARY_OP)
        gate_op = self._op(self._BINARY_OP)

        class ChainedGate(nn.Module):
            def __init__(self):
                super().__init__()
                self._act = act_op
                self._gate = gate_op

            def forward(self, x, y):
                return self._gate(self._act(x), y)

        inputs = [torch.randn(16, 256, device="cuda"), torch.randn(16, 256, device="cuda")]
        ref_fn = lambda x, y: self._BINARY_REF(self._UNARY_REF(x), y)
        _check(self, ChainedGate().eval().cuda(), inputs, ref_fn,
               f"{self._NS}::{self._BINARY_OP}")


# ---------------------------------------------------------------------------
# Mixin: multi-config tactic selection (Triton + CuTile only)
# ---------------------------------------------------------------------------

class _MultiConfigTests:
    """Two additional tests for backends that register multi-config variants.

    TRT's autotuner benchmarks each tile-size configuration
    (e.g. BLOCK_SIZE ∈ {64, 128, 256}) and picks the fastest tactic for the
    target GPU at engine-build time.  Correctness against the reference is
    still verified.

    Multi-config op names follow the convention '<base_op>_mc'.
    Shape [16, 512] gives all three tile sizes non-trivial work.
    """

    def test_multi_config_unary(self):
        """Unary op with 3 tile-size configs; TRT picks the best tactic."""
        op_name = f"{self._UNARY_OP}_mc"
        m = _UnaryOp(self._op(op_name)).eval().cuda()
        inputs = [torch.randn(16, 512, device="cuda")]
        _check(self, m, inputs, self._UNARY_REF, f"{self._NS}::{op_name}")

    def test_multi_config_binary(self):
        """Binary op with 3 tile-size configs; TRT picks the best tactic."""
        op_name = f"{self._BINARY_OP}_mc"
        m = _BinaryOp(self._op(op_name)).eval().cuda()
        inputs = [torch.randn(16, 512, device="cuda"), torch.randn(16, 512, device="cuda")]
        _check(self, m, inputs, self._BINARY_REF, f"{self._NS}::{op_name}")


# ---------------------------------------------------------------------------
# Concrete test classes — one per backend
# ---------------------------------------------------------------------------

class TestTritonE2E(_MultiConfigTests, _BackendE2ETests, unittest.TestCase):
    """Triton backend: SiLU activation + SwiGLU gating (6 tests).

    SiLU and SwiGLU are the canonical activation ops in the LLaMA / Mistral
    family.  The SwiGLU FFN block tested here is structurally identical to
    what ships in those models.
    """

    _NS = "torchtrt_e2e_triton"
    _UNARY_OP = "silu"
    _BINARY_OP = "swiglu"
    _UNARY_REF = staticmethod(_ref_silu)
    _BINARY_REF = staticmethod(_ref_swiglu)


class TestCuTileE2E(_MultiConfigTests, _BackendE2ETests, unittest.TestCase):
    """CuTile backend: ReLU activation + ReGLU gating (6 tests).

    ReGLU (relu(gate) · up) is a simpler gated variant used in ablation
    studies and in quantized deployments where ReLU sparsity is desirable.
    """

    _NS = "torchtrt_e2e_cutile"
    _UNARY_OP = "relu"
    _BINARY_OP = "reglu"
    _UNARY_REF = staticmethod(_ref_relu)
    _BINARY_REF = staticmethod(_ref_reglu)


class TestCuTeDSLE2E(_BackendE2ETests, unittest.TestCase):
    """CuTeDSL backend: SiLU activation + Hadamard gating (4 tests).

    SiLU is implemented via cute.arch.exp (lowers to CUDA expf intrinsic).
    Hadamard product models the element-wise combination step in attention
    masking, LoRA weight updates, and gated linear units.
    """

    _NS = "torchtrt_e2e_cutedsl"
    _UNARY_OP = "silu"
    _BINARY_OP = "hadamard"
    _UNARY_REF = staticmethod(_ref_silu)
    _BINARY_REF = staticmethod(_ref_hadamard)


# ---------------------------------------------------------------------------
# Cross-backend tests: multiple PluginV3 ops from different backends in one engine
# ---------------------------------------------------------------------------

# Convenience references to ops registered above.
_triton_silu   = lambda: torch.ops.torchtrt_e2e_triton.silu.default
_triton_swiglu = lambda: torch.ops.torchtrt_e2e_triton.swiglu.default
_cutile_relu   = lambda: torch.ops.torchtrt_e2e_cutile.relu.default
_cutile_reglu  = lambda: torch.ops.torchtrt_e2e_cutile.reglu.default
_cutedsl_silu  = lambda: torch.ops.torchtrt_e2e_cutedsl.silu.default
_cutedsl_had   = lambda: torch.ops.torchtrt_e2e_cutedsl.hadamard.default



class TestCrossBackendE2E(unittest.TestCase):
    """Cross-backend tests: two or three PluginV3 ops from different backends in one engine.

    All shapes are LLM-domain [batch=8, hidden=512].  Each test uses
    _check_multi_op to assert that every named PluginV3 layer is present in the
    single compiled TRT engine.
    """

    def test_triton_then_cutile_unary(self):
        """Triton SiLU → CuTile ReLU: two sequential unary activations."""
        class Model(nn.Module):
            def forward(self, x):
                return _cutile_relu()(_triton_silu()(x))

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x: _ref_relu(_ref_silu(x))
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_triton::silu", "torchtrt_e2e_cutile::relu"])

    def test_triton_then_cutedsl_unary(self):
        """Triton SiLU → CuTeDSL SiLU: two sequential SiLU activations, different backends."""
        class Model(nn.Module):
            def forward(self, x):
                return _cutedsl_silu()(_triton_silu()(x))

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x: _ref_silu(_ref_silu(x))
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_triton::silu", "torchtrt_e2e_cutedsl::silu"])

    def test_cutile_then_cutedsl_unary(self):
        """CuTile ReLU → CuTeDSL SiLU: two sequential unary ops, different backends."""
        class Model(nn.Module):
            def forward(self, x):
                return _cutedsl_silu()(_cutile_relu()(x))

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x: _ref_silu(_ref_relu(x))
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_cutile::relu", "torchtrt_e2e_cutedsl::silu"])

    def test_triton_unary_then_cutile_binary(self):
        """Triton SiLU activation followed by CuTile ReGLU gating.

        Models the pattern: activated_gate = silu(x); out = reglu(activated_gate, up)
        where the gate branch passes through a Triton activation before the CuTile
        binary gating op.
        """
        class Model(nn.Module):
            def forward(self, x, y):
                return _cutile_reglu()(_triton_silu()(x), y)

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x, y: _ref_reglu(_ref_silu(x), y)
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_triton::silu", "torchtrt_e2e_cutile::reglu"])

    def test_cutile_unary_then_triton_binary(self):
        """CuTile ReLU activation followed by Triton SwiGLU gating."""
        class Model(nn.Module):
            def forward(self, x, y):
                return _triton_swiglu()(_cutile_relu()(x), y)

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x, y: _ref_swiglu(_ref_relu(x), y)
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_cutile::relu", "torchtrt_e2e_triton::swiglu"])

    def test_cutedsl_unary_then_triton_binary(self):
        """CuTeDSL SiLU activation followed by Triton SwiGLU gating."""
        class Model(nn.Module):
            def forward(self, x, y):
                return _triton_swiglu()(_cutedsl_silu()(x), y)

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x, y: _ref_swiglu(_ref_silu(x), y)
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_cutedsl::silu", "torchtrt_e2e_triton::swiglu"])

    def test_triton_swiglu_then_cutedsl_hadamard(self):
        """Triton SwiGLU gating followed by CuTeDSL Hadamard masking.

        Models a two-stage gate computation: swiglu merges gate + up, then
        hadamard applies a learned mask (e.g. LoRA adapter residual scaling).
        """
        class Model(nn.Module):
            def forward(self, gate, up, mask):
                h = _triton_swiglu()(gate, up)
                return _cutedsl_had()(h, mask)

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda gate, up, mask: _ref_hadamard(_ref_swiglu(gate, up), mask)
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_triton::swiglu", "torchtrt_e2e_cutedsl::hadamard"])

    def test_cutile_reglu_then_cutedsl_hadamard(self):
        """CuTile ReGLU gating followed by CuTeDSL Hadamard masking."""
        class Model(nn.Module):
            def forward(self, gate, up, mask):
                h = _cutile_reglu()(gate, up)
                return _cutedsl_had()(h, mask)

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda"),
                  torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda gate, up, mask: _ref_hadamard(_ref_reglu(gate, up), mask)
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_cutile::reglu", "torchtrt_e2e_cutedsl::hadamard"])

    def test_all_three_unary_chain(self):
        """Triton SiLU → CuTile ReLU → CuTeDSL SiLU: three backends in one engine.

        Verifies that three PluginV3 layers from three different backends all
        coexist in a single TRT engine and that the data flows correctly through
        each activation in sequence.
        """
        class Model(nn.Module):
            def forward(self, x):
                x = _triton_silu()(x)
                x = _cutile_relu()(x)
                x = _cutedsl_silu()(x)
                return x

        inputs = [torch.randn(_LLM_B, _LLM_H, device="cuda")]
        ref_fn = lambda x: _ref_silu(_ref_relu(_ref_silu(x)))
        _check_multi_op(self, Model().eval().cuda(), inputs, ref_fn,
                        ["torchtrt_e2e_triton::silu",
                         "torchtrt_e2e_cutile::relu",
                         "torchtrt_e2e_cutedsl::silu"])


# ---------------------------------------------------------------------------
# Plugin attrs E2E
# ---------------------------------------------------------------------------

class TestAttrsE2E(unittest.TestCase):
    """Verify attrs flow from tta.custom_plugin(SCALE=2.0) through TRT to the kernel.

    The op 'torchtrt_e2e_triton::silu_scaled' is compiled with SCALE=2.0
    baked in as a tl.constexpr.  Output must equal 2.0 * silu(x).
    """

    def test_triton_attrs_scale_factor(self):
        """SCALE attr is baked into PTX as constexpr; output is 2·x·σ(x)."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_scaled.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: 2.0 * x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_scaled")


# ---------------------------------------------------------------------------
# Multiple outputs E2E
# ---------------------------------------------------------------------------

class TestMultiOutputE2E(unittest.TestCase):
    """Verify a PluginV3 op with two output tensors compiles and runs correctly.

    The op 'torchtrt_e2e_triton::dual_activation' returns (silu(x), relu(x)).
    The test combines them as silu(x) + relu(x) for a single-output comparison.
    """

    def test_triton_dual_output_plugin(self):
        """Dual-output PluginV3: engine has one layer, two ITensor outputs."""
        op = torch.ops.torchtrt_e2e_triton.dual_activation.default

        class DualOutputModel(nn.Module):
            def forward(self, x):
                silu_out, relu_out = op(x)
                return silu_out + relu_out

        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x) + torch.relu(x)
        _check(self, DualOutputModel().eval().cuda(), inputs, ref_fn,
               "torchtrt_e2e_triton::dual_activation")


# ---------------------------------------------------------------------------
# Non-float32 dtype E2E
# ---------------------------------------------------------------------------

class TestDtypeE2E(unittest.TestCase):
    """Verify custom plugins handle non-float32 tensor dtypes end-to-end."""

    def test_triton_silu_float16(self):
        """SiLU op with float16 input/output using a dedicated fp16-safe kernel."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_f16.default).eval().cuda()
        inputs = [torch.randn(8, 512, dtype=torch.float16, device="cuda")]
        ref_fn = lambda x: (x.float() * torch.sigmoid(x.float())).half()
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_f16",
               rtol=1e-3, atol=1e-3)

    def test_cutile_relu_float16(self):
        """CuTile ReLU with float16 input/output."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_cutile.relu.default).eval().cuda()
        inputs = [torch.randn(8, 512, dtype=torch.float16, device="cuda")]
        ref_fn = lambda x: torch.relu(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_cutile::relu",
               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# supports_dynamic_shapes=False E2E
# ---------------------------------------------------------------------------

class TestStaticShapeE2E(unittest.TestCase):
    """Verify an op registered with supports_dynamic_shapes=False compiles correctly."""

    def test_static_shape_engine(self):
        """Fixed-shape input compiles to a valid TRT engine."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_static.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_static")

    def test_static_shape_different_batch(self):
        """Different fixed-shape input also compiles independently."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_static.default).eval().cuda()
        inputs = [torch.randn(4, 256, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_static")


# ---------------------------------------------------------------------------
# Tensor format E2E
# ---------------------------------------------------------------------------

class TestTensorFormatE2E(unittest.TestCase):
    """Verify explicit input_formats/output_formats flow through to TRT autotune."""

    def test_explicit_linear_format(self):
        """Explicit LINEAR format spec compiles and produces correct output."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_linear_fmt.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_linear_fmt")

    def test_explicit_linear_format_with_llm_shape(self):
        """Explicit LINEAR format spec at LLM scale [batch=8, hidden=512]."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_linear_fmt.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_linear_fmt")


# ---------------------------------------------------------------------------
# Weights E2E
# ---------------------------------------------------------------------------

class TestWeightsE2E(unittest.TestCase):
    """Verify tensor weights injected via custom_plugin(W=tensor) are baked into
    the engine as trt.add_constant layers and reach the kernel correctly.

    Engine check: the output equals W * silu(x) where W is the module-level
    _W_COLUMN_SCALE tensor (all-3.0).  A plain silu(x) reference would differ
    by 3×, so any mismatch in weight injection is immediately caught by the
    accuracy assertion.
    """

    def test_triton_column_scale_weight(self):
        """Weight tensor W[j] scales column j of silu(x); engine bakes W as constant."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.weighted_silu.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        W_dev = _W_COLUMN_SCALE.cuda()
        ref_fn = lambda x: W_dev * x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::weighted_silu")


# ---------------------------------------------------------------------------
# BFloat16 dtype E2E
# ---------------------------------------------------------------------------

class TestBF16DtypeE2E(unittest.TestCase):
    """Verify custom plugins compile and produce numerically correct results
    for bfloat16 inputs.  Two engine-level checks per test:
      1. The TRT engine contains a PluginV3 layer for the op (via _check).
      2. The output tensor dtype is torch.bfloat16 (explicit assertion).
    """

    def test_triton_silu_bfloat16(self):
        """BF16 Triton SiLU: engine runs in bf16; output dtype is bfloat16."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_bf16.default).eval().cuda()
        inputs = [torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")]
        ref_fn = lambda x: (x.float() * torch.sigmoid(x.float())).bfloat16()
        compiled = _trt_compile(m, inputs)
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_triton::silu_bf16"))
        trt_out = compiled(*inputs)
        # Engine-level dtype check: output must be bf16, not fp32.
        self.assertEqual(trt_out.dtype, torch.bfloat16,
                         "Expected engine output dtype to be bfloat16")
        torch.testing.assert_close(trt_out, ref_fn(*inputs), rtol=2e-2, atol=2e-2)

    def test_cutile_relu_bfloat16(self):
        """BF16 CuTile ReLU: engine runs in bf16; output dtype is bfloat16."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_cutile.relu.default).eval().cuda()
        inputs = [torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")]
        ref_fn = lambda x: torch.relu(x)
        compiled = _trt_compile(m, inputs)
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_cutile::relu"))
        trt_out = compiled(*inputs)
        self.assertEqual(trt_out.dtype, torch.bfloat16,
                         "Expected engine output dtype to be bfloat16")
        torch.testing.assert_close(trt_out, ref_fn(*inputs), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# 3D input E2E
# ---------------------------------------------------------------------------

class Test3DInputE2E(unittest.TestCase):
    """Verify custom plugins compile and run correctly for rank-3 inputs
    [batch, seq_len, hidden].

    Engine checks:
      1. PluginV3 layer present (via _engine_has_pluginv3_layer).
      2. Output shape equals input shape (rank preserved through engine).
    """

    def test_triton_silu_3d_input(self):
        """Triton flat-SiLU on [2, 16, 256]: output shape and values correct."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_flat.default).eval().cuda()
        inputs = [torch.randn(2, 16, 256, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        compiled = _trt_compile(m, inputs)
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_triton::silu_flat"))
        trt_out = compiled(*inputs)
        # Engine-level shape check: rank-3 shape must be preserved end-to-end.
        self.assertEqual(tuple(trt_out.shape), tuple(inputs[0].shape),
                         f"Expected output shape {inputs[0].shape}, got {trt_out.shape}")
        torch.testing.assert_close(trt_out, ref_fn(*inputs), rtol=1e-3, atol=1e-3)

    def test_cutile_relu_3d_input(self):
        """CuTile ReLU on [2, 16, 256]: output shape and values correct."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_cutile.relu.default).eval().cuda()
        inputs = [torch.randn(2, 16, 256, device="cuda")]
        ref_fn = lambda x: torch.relu(x)
        compiled = _trt_compile(m, inputs)
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_cutile::relu"))
        trt_out = compiled(*inputs)
        self.assertEqual(tuple(trt_out.shape), tuple(inputs[0].shape),
                         f"Expected output shape {inputs[0].shape}, got {trt_out.shape}")
        torch.testing.assert_close(trt_out, ref_fn(*inputs), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Multiple attrs E2E
# ---------------------------------------------------------------------------

class TestMultiAttrsE2E(unittest.TestCase):
    """Verify that multiple scalar attrs (SCALE, BIAS) are all baked into PTX
    as separate tl.constexpr values.

    Engine check: output equals SCALE * silu(x) + BIAS with both values set
    independently.  A single-attr kernel with only SCALE would produce
    SCALE * silu(x) without the BIAS offset, failing the assertion.
    """

    def test_triton_scale_and_bias_attrs(self):
        """SCALE=3.0 and BIAS=1.0 both baked as constexprs; output verified."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu_scale_bias.default).eval().cuda()
        inputs = [torch.randn(8, 512, device="cuda")]
        ref_fn = lambda x: 3.0 * x * torch.sigmoid(x) + 1.0
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu_scale_bias")


# ---------------------------------------------------------------------------
# Multi-output + dynamic batch E2E
# ---------------------------------------------------------------------------

class TestMultiOutputDynamicE2E(unittest.TestCase):
    """Verify a multi-output plugin compiles with dynamic batch and produces
    correct outputs at three different batch sizes.

    Engine checks per batch size:
      1. PluginV3 layer present (verified at compile time via single compile).
      2. Both silu and relu outputs numerically correct (two allclose checks).
    """

    def test_dual_output_dynamic_batch(self):
        """dual_activation with dynamic batch: both outputs correct at 1, 8, 32."""
        op = torch.ops.torchtrt_e2e_triton.dual_activation.default

        class DualModel(nn.Module):
            def forward(self, x):
                silu_out, relu_out = op(x)
                return silu_out + relu_out

        torch._dynamo.reset()
        compiled = torch_tensorrt.compile(
            DualModel().eval().cuda(),
            inputs=[torch_tensorrt.Input(
                min_shape=(1, 512), opt_shape=(8, 512), max_shape=(32, 512),
                dtype=torch.float32,
            )],
            min_block_size=1,
        )
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_triton::dual_activation"))
        for batch in (1, 8, 32):
            x = torch.randn(batch, 512, device="cuda")
            trt_out = compiled(x)
            ref = x * torch.sigmoid(x) + torch.relu(x)
            torch.testing.assert_close(trt_out, ref, rtol=1e-3, atol=1e-3,
                                       msg=f"Mismatch at batch={batch}")


# ---------------------------------------------------------------------------
# Dynamic hidden dimension E2E
# ---------------------------------------------------------------------------

class TestDynamicHiddenE2E(unittest.TestCase):
    """Verify a plugin compiled with a dynamic hidden dimension (dim-1, not dim-0)
    produces correct outputs across the full hidden range.

    Engine check: PluginV3 present + outputs correct at hidden = 128, 256, 512.
    This is distinct from test_dynamic_batch (which varies dim-0 only).
    """

    def test_triton_dynamic_hidden_dim(self):
        """SiLU with dynamic hidden; correct at min=128, opt=256, max=512."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu.default).eval().cuda()
        torch._dynamo.reset()
        compiled = torch_tensorrt.compile(
            m,
            inputs=[torch_tensorrt.Input(
                min_shape=(8, 128), opt_shape=(8, 256), max_shape=(8, 512),
                dtype=torch.float32,
            )],
            min_block_size=1,
        )
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_triton::silu"))
        for hidden in (128, 256, 512):
            x = torch.randn(8, hidden, device="cuda")
            trt_out = compiled(x)
            torch.testing.assert_close(trt_out, x * torch.sigmoid(x),
                                       rtol=1e-3, atol=1e-3,
                                       msg=f"Mismatch at hidden={hidden}")


# ---------------------------------------------------------------------------
# Production scale E2E
# ---------------------------------------------------------------------------

class TestLargeShapeE2E(unittest.TestCase):
    """Verify the plugin pipeline handles production-scale LLM shapes.

    [32, 4096] matches GPT-2 large / LLaMA-7B hidden_size=4096, batch=32.
    Engine check: PluginV3 present + accuracy within tolerance at this scale.
    """

    def test_production_scale(self):
        """Triton SiLU at [32, 4096] — production LLM inference scale."""
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.silu.default).eval().cuda()
        inputs = [torch.randn(32, 4096, device="cuda")]
        ref_fn = lambda x: x * torch.sigmoid(x)
        _check(self, m, inputs, ref_fn, "torchtrt_e2e_triton::silu")


# ---------------------------------------------------------------------------
# Cross-backend + dynamic batch E2E
# ---------------------------------------------------------------------------

class TestCrossBackendDynamicE2E(unittest.TestCase):
    """Verify that mixing Triton and CuTile plugins in one engine still works
    under dynamic batch shapes.

    Engine checks:
      1. Both PluginV3 layers present in a single engine.
      2. Outputs correct at batch = 1, 8, and 32.
    """

    def test_dynamic_batch_cross_backend(self):
        """Triton SiLU → CuTile ReLU in one engine; correct at 3 batch sizes."""
        class ChainModel(nn.Module):
            def forward(self, x):
                x = torch.ops.torchtrt_e2e_triton.silu.default(x)
                return torch.ops.torchtrt_e2e_cutile.relu.default(x)

        torch._dynamo.reset()
        compiled = torch_tensorrt.compile(
            ChainModel().eval().cuda(),
            inputs=[torch_tensorrt.Input(
                min_shape=(1, 256), opt_shape=(8, 256), max_shape=(32, 256),
                dtype=torch.float32,
            )],
            min_block_size=1,
        )
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1, "Expected both plugins in one TRT engine")
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_triton::silu"))
        self.assertTrue(_engine_has_pluginv3_layer(engines[0].engine,
                                                    "torchtrt_e2e_cutile::relu"))
        for batch in (1, 8, 32):
            x = torch.randn(batch, 256, device="cuda")
            trt_out = compiled(x)
            ref = torch.relu(x * torch.sigmoid(x))
            torch.testing.assert_close(trt_out, ref, rtol=1e-3, atol=1e-3,
                                       msg=f"Mismatch at batch={batch}")


# ---------------------------------------------------------------------------
# Non-LINEAR tensor format E2E
# ---------------------------------------------------------------------------


class TestNonLinearFormatsE2E(unittest.TestCase):
    """Verify that non-LINEAR ``input_formats`` / ``output_formats`` on a
    TritonSpec are correctly propagated through the QDP autotune registration
    and result in TRT inserting format-conversion (reformat) layers around the
    PluginV3 layer in the compiled engine.

    The observable signal for format negotiation is the presence of
    ``"Reformatting CopyNode"`` layer names in the engine inspector output.
    When a plugin declares CHW32, TRT inserts:
      - a LINEAR → CHW32 reformat before the plugin
      - a CHW32 → LINEAR reformat after the plugin
    so the engine has 3 layers total instead of 1.

    Input shape [4, 32, 8, 8] is used because CHW32 requires the channel
    dimension (dim 1) to be a multiple of 32.

    The flat-indexed ReLU kernel (_triton_launch_flat_relu) uses contiguous
    pointer arithmetic so it is correct for any memory layout, including CHW32.
    """

    _OP = "torchtrt_e2e_triton::relu_chw32"
    _SHAPE = (4, 32, 8, 8)

    def _compile(self):
        m = _UnaryOp(torch.ops.torchtrt_e2e_triton.relu_chw32.default).eval().cuda()
        torch._dynamo.reset()
        return torch_tensorrt.compile(
            m,
            inputs=[torch.zeros(self._SHAPE, device="cuda")],
            min_block_size=1,
        )

    def test_pluginv3_present(self):
        """CHW32 plugin compiles into a TRT engine with a PluginV3 layer."""
        compiled = self._compile()
        engines = _get_trt_engines(compiled)
        self.assertEqual(len(engines), 1)
        self.assertTrue(
            _engine_has_pluginv3_layer(engines[0].engine, self._OP),
            f"Expected PluginV3 layer for {self._OP!r} in engine",
        )

    def test_reformat_nodes_present(self):
        """TRT inserts reformatting nodes around the CHW32 plugin.

        This confirms the CHW32 format token from ``input_formats`` was
        correctly propagated into the ``AutoTuneCombination`` and TRT
        negotiated the non-LINEAR format for the plugin layer.

        When TRT selects CHW32, it inserts format-conversion layers before and
        after the plugin.  These may appear as ``"Reformatting CopyNode"``
        entries in the raw inspector output, or as Myelin-fused ``__mye*`` /
        ``__myl*`` nodes after graph optimization.  Either way the engine has
        more than one layer, unlike a LINEAR plugin which needs no reformats.
        """
        compiled = self._compile()
        engines = _get_trt_engines(compiled)
        layer_names = _engine_all_layer_names(engines[0].engine)
        # Find layers that are NOT the plugin itself — these are the reformat
        # (or Myelin-fused reformat) nodes inserted by TRT for CHW32.
        non_plugin = [
            ln for ln in layer_names
            if "relu_chw32" not in ln
        ]
        self.assertGreater(
            len(non_plugin), 0,
            f"Expected extra reformat layers for CHW32 format negotiation, "
            f"but found none. Layer names: {layer_names}",
        )

    def test_output_correct(self):
        """Output values are correct despite CHW32 format repack."""
        compiled = self._compile()
        x = torch.randn(self._SHAPE, device="cuda")
        trt_out = compiled(x)
        ref = torch.relu(x)
        torch.testing.assert_close(trt_out, ref, rtol=1e-5, atol=1e-5)

