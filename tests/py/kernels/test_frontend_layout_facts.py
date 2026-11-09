# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CuTe DSL driver ISA ceiling and pointer alignment; cuTile AOT layout facts.

Self-contained: CPU cases need only the Python packages; GPU cases skip without
CUDA, a QDP-capable TensorRT, or the DSL toolchain.
"""

import dataclasses
import re
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import torch_tensorrt
from torch_tensorrt.kernels import _common, _cutedsl, _cutile

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
skip_no_qdp = pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="TensorRT QDP plugin not available",
)


def _cutile_toolchain_reason():
    try:
        import cuda.tile  # noqa: F401
        from cuda.tile.compilation import export_kernel  # noqa: F401

        _cutile.validate_cutile_toolchain("layout facts tests")
    except (ImportError, RuntimeError) as error:
        return str(error)
    return None


def _mem_ops(ptx: str) -> dict:
    ops: dict = {}
    for op in re.findall(r"\b((?:ld|st)\.global(?:\.[a-z0-9:]+)*)", ptx):
        ops[op] = ops.get(op, 0) + 1
    return ops


# ---- CuTe DSL: driver PTX ISA ceiling and element-size alignment ----


@pytest.mark.parametrize(
    "version,expected",
    [
        (12080, 87),
        (13020, 92),
        (13030, 93),
        (13040, 94),
        (13090, 99),
        # PTX ISA 9.10 has no two-digit encoding; 100 would mean ISA 10.0.
        (13100, None),
        (13120, None),
        (14000, None),
    ],
)
def test_cuda13_drivers_newer_than_the_table_keep_a_ceiling(
    version, expected, monkeypatch
):
    driver = pytest.importorskip("cuda.bindings.driver")
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: (0, version))
    assert _cutedsl._driver_ptx_version() == expected


@pytest.mark.parametrize(
    "alignment,expected",
    [
        (None, {"x": 4, "h": 2, "out": 2}),
        ({"x": 16}, {"x": 16, "h": 2, "out": 2}),
        (16, {"x": 16, "h": 16, "out": 16}),
        (1, {"x": 1, "h": 1, "out": 1}),
    ],
)
def test_pointer_alignment_defaults_to_the_element_size(alignment, expected):
    signature = {"x": "*fp32", "h": "*bf16", "n": "i32", "out": "*fp16"}
    layout = _common.analyze_signature(signature, (2, 1), frontend="cutedsl_op")
    assert _cutedsl.normalize_pointer_alignment("ns::op", layout, alignment) == expected


@skip_no_cuda
@skip_no_qdp
def test_cutedsl_op_default_alignment_vectorizes_float32_and_runs_in_an_engine():
    """autovec_copy on fp32 did not compile at the old default alignment of 1."""
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    import torch_tensorrt.kernels as ttk

    @cute.kernel
    def double(x, out, VEC: cutlass.Constexpr, THREADS: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        offset = cute.assume((bidx * THREADS + tidx) * VEC, divby=VEC)
        src = cute.make_tensor(x + offset, cute.make_layout(VEC))
        dst = cute.make_tensor(out + offset, cute.make_layout(VEC))
        regs = cute.make_rmem_tensor(VEC, cutlass.Float32)
        cute.autovec_copy(src, regs)
        regs.store(regs.load() * 2.0)
        cute.autovec_copy(regs, dst)

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    op_name = "frontend_fixes::cutedsl_double"
    if not hasattr(torch.ops.frontend_fixes, "cutedsl_double"):
        ttk.cutedsl_op(
            op_name,
            double,
            {"x": "*fp32", "out": "*fp32"},
            meta,
            grid=lambda inputs, outputs: (4,),
            block_size=128,
            constants={"VEC": 4, "THREADS": 128},
        )

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.frontend_fixes.cutedsl_double(x)

    x = torch.randn(4 * 128 * 4, device="cuda")
    compiled = torch_tensorrt.compile(
        Model().eval().cuda(),
        ir="dynamo",
        inputs=[x],
        min_block_size=1,
        strict=True,
        require_full_compilation=True,
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), x * 2)


# ---- cuTile: AOT layout facts ----


def _layout(signature, names=("x",)):
    return _cutile.validate_cutile_config(
        "ns::op",
        signature,
        {},
        (len(names), len(signature) - len(names)),
        input_names=names,
    )


def test_derived_launch_only_assumes_the_unit_last_stride():
    layout = _layout({"x": "bf16[2]", "out": "fp32"})
    assert _cutile.array_layout_facts("ns::op", layout) == [
        {"stride_constant": (None, 1)},
        {"stride_constant": (1,)},
    ]
    # A custom aot_fn supplies its own strides, so nothing is assumed about them.
    facts = _cutile.array_layout_facts("ns::op", layout, derived_launch=False)
    assert facts == [{}, {}]


@pytest.mark.parametrize(
    "pointer_alignment, base", [(None, None), (1, None), (8, 8), (16, 16)]
)
def test_pointer_alignment_is_an_explicit_promise(pointer_alignment, base):
    # The default, None, like 1, leaves cuTile's "no assumption" in place.
    layout = _layout({"x": "fp32", "out": "fp32"})
    facts = _cutile.array_layout_facts(
        "ns::op", layout, pointer_alignment=pointer_alignment
    )
    assert [fact.get("base_addr_divisible_by") for fact in facts] == [base, base]


def test_static_shapes_add_divisibility_like_cutile_jax():
    layout = _layout({"x": "bf16[2]", "w": "bf16", "out": "fp32"}, names=("x", "w"))
    facts = _cutile.array_layout_facts(
        "ns::op",
        layout,
        static_shapes={"x": (4096, 2048), "w": (3, 5), "out": (64, 1024)},
    )
    assert facts[0] == {
        "stride_constant": (None, 1),
        "stride_divisible_by": (8, 1),
        "shape_divisible_by": (16, 16),
    }
    # A rank-1 array is the flattened view: one extent of 15 elements.
    assert facts[1] == {
        "stride_constant": (1,),
        "stride_divisible_by": (1,),
        "shape_divisible_by": (1,),
    }
    assert facts[2]["shape_divisible_by"] == (16,)


def test_static_shapes_keep_unproven_divisibility_at_one():
    layout = _layout({"x": "fp32[3]", "out": "fp32[3]"})
    facts = _cutile.array_layout_facts(
        "ns::op", layout, static_shapes={"x": (2, 3, 6), "out": (5, 1, 8)}
    )
    assert facts[0]["stride_divisible_by"] == (1, 1, 1)
    assert facts[0]["shape_divisible_by"] == (1, 1, 1)
    # Strides of 8, 8 and 1 float32 elements: the leading two step 32 bytes.
    assert facts[1]["stride_constant"] == (None, None, 1)
    assert facts[1]["stride_divisible_by"] == (4, 4, 1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(pointer_alignment=0), "power of two"),
        (dict(pointer_alignment=24), "power of two"),
        (dict(pointer_alignment=512), "power of two"),
        (dict(pointer_alignment=True), "power of two"),
        (dict(static_shapes={"x": (4,)}), "exactly the signature's arrays"),
        (dict(static_shapes={"x": (4,), "out": "4"}), "non-negative ints"),
        (dict(static_shapes={"x": (4,), "out": (4, -1)}), "non-negative ints"),
        (dict(static_shapes={"x": (4,), "out": (4,)}, derived_launch=False), "aot_fn"),
    ],
)
def test_invalid_layout_facts_are_rejected(kwargs, match):
    layout = _layout({"x": "fp32", "out": "fp32"})
    with pytest.raises(ValueError, match=match):
        _cutile.array_layout_facts("ns::op", layout, **kwargs)


def test_rank_must_match_a_static_shape_beyond_rank_one():
    layout = _layout({"x": "fp32[2]", "out": "fp32"})
    with pytest.raises(ValueError, match="2 dimension"):
        _cutile.array_layout_facts(
            "ns::op", layout, static_shapes={"x": (8,), "out": (8,)}
        )


def _node(*shapes):
    *inputs, output = shapes
    value = lambda shape: torch.empty(shape, device="meta")  # noqa: E731
    return SimpleNamespace(
        args=tuple(SimpleNamespace(meta={"val": value(shape)}) for shape in inputs),
        meta={"val": value(output)},
    )


def test_static_shape_validator_declines_other_shapes():
    layout = _layout({"x": "fp32", "out": "fp32"})
    validate = _cutile.make_dtype_capability_validator(
        "ns::op", layout, static_shapes={"x": (4, 8), "out": (4, 8)}
    )
    assert validate(_node((4, 8), (4, 8)))
    assert not validate(_node((2, 8), (2, 8)))
    unconstrained = _cutile.make_dtype_capability_validator("ns::op", layout)
    assert unconstrained(_node((2, 8), (2, 8)))


def _view_operations():
    operations = _cutile._tma_hints()
    if operations is None:
        pytest.skip("cuda-tile without the view load/store hook")
    return operations


def test_tma_hook_is_scoped_to_the_export():
    operations = _view_operations()
    originals = [operation.generate_bytecode for operation in operations]
    with pytest.raises(RuntimeError):
        with _cutile._without_tma(operations):
            for operation, original in zip(operations, originals):
                assert operation.generate_bytecode is not original
            raise RuntimeError("export failed")
    assert [operation.generate_bytecode for operation in operations] == originals


def test_tma_hook_leaves_pointer_loads_and_stores_alone():
    """tileiras rejects allow_tma on the pointer loads and stores of ct.gather
    and ct.scatter, which never use TMA."""
    operations = _view_operations()
    from cuda.tile._ir import ops
    from cuda.tile._ir2bytecode import BytecodeContext

    assert operations == (ops.TileLoad, ops.TileStore)

    def untouched():
        return (
            ops.LoadPointer.generate_bytecode,
            ops.StorePointer.generate_bytecode,
            BytecodeContext.load_store_hints,
        )

    before = untouched()
    with _cutile._without_tma(operations):
        assert untouched() == before


@pytest.mark.parametrize("change", ["no allow_tma", "new signature"])
def test_tma_hook_is_missing_when_view_operations_change(monkeypatch, change):
    _view_operations()
    from cuda.tile._ir import ops

    def generate_bytecode(self, ctx):
        pass

    def generate_bytecode_with_options(self, ctx, options):
        pass

    if change == "no allow_tma":
        fields, method = ["latency"], generate_bytecode
    else:
        fields, method = ["latency", "allow_tma"], generate_bytecode_with_options
    moved = dataclasses.make_dataclass(
        "TileLoad", fields, namespace={"generate_bytecode": method}
    )
    monkeypatch.setattr(ops, "TileLoad", moved)
    assert _cutile._tma_hints() is None


class _ViewOperation:
    """Stands in for one of cuTile's view load or store operations."""

    def __init__(self, allow_tma):
        self.allow_tma = allow_tma

    def generate_bytecode(self, ctx):
        return ctx, self.allow_tma


def test_tma_hook_leaves_each_operation_as_it_was():
    """An operation belongs to the kernel's IR; only its bytecode changes."""

    class Failing(_ViewOperation):
        def generate_bytecode(self, ctx):
            raise RuntimeError(f"allow_tma={self.allow_tma}")

    operation = type("TileLoad", (_ViewOperation,), {})
    with _cutile._without_tma((operation, Failing)):
        load = operation(None)
        assert load.generate_bytecode("ctx") == ("ctx", False)
        assert load.allow_tma is None
        failing = Failing(True)
        with pytest.raises(RuntimeError, match="allow_tma=False"):
            failing.generate_bytecode("ctx")
        assert failing.allow_tma is True
    assert operation(True).generate_bytecode("ctx") == ("ctx", True)


def _run_threads(*targets):
    # Daemon threads, so that a scope that deadlocks fails the test instead of
    # keeping the interpreter from exiting.
    threads = [threading.Thread(target=target, daemon=True) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a TMA scope hung"


def test_overlapping_tma_scopes_in_threads_restore_the_hook():
    """A scope that read the hook before taking the lock restored the other
    scope's patch, which left TMA disabled for every later compilation."""
    operation = type("TileLoad", (_ViewOperation,), {})
    original = operation.generate_bytecode
    first_patched, second_entering = threading.Event(), threading.Event()
    hints = []

    def first():
        with _cutile._without_tma((operation,)):
            first_patched.set()
            second_entering.wait(timeout=5)
            # Give the second scope time to read the hook while it is patched.
            time.sleep(0.2)
            hints.append(operation(True).generate_bytecode(1))

    def second():
        first_patched.wait(timeout=5)
        second_entering.set()
        with _cutile._without_tma((operation,)):
            hints.append(operation(True).generate_bytecode(2))

    _run_threads(first, second)
    assert hints == [(1, False), (2, False)]
    assert operation.generate_bytecode is original
    assert operation(True).generate_bytecode(3) == (3, True)


def test_nested_tma_scopes_in_one_thread_unwind_in_order():
    operation = type("TileStore", (_ViewOperation,), {})
    original = operation.generate_bytecode
    hints = []

    def nested():
        with _cutile._without_tma((operation,)):
            with _cutile._without_tma((operation,)):
                hints.append(operation(True).generate_bytecode(1))
            hints.append(operation(True).generate_bytecode(2))

    _run_threads(nested)
    assert hints == [(1, False), (2, False)]
    assert operation.generate_bytecode is original


def test_missing_tma_hook_drops_only_the_alignment_promise(monkeypatch):
    pytest.importorskip("cuda.tile")
    monkeypatch.setattr(_cutile, "_tma_hints", lambda: None)
    monkeypatch.setattr(_cutile, "validate_cutile_toolchain", lambda name: (13, 2))
    captured = {}

    def export(kernel, signatures, buffer, **kwargs):
        captured["arrays"] = signatures[0].parameters
        raise FileNotFoundError("stop after building the signature")

    monkeypatch.setattr("cuda.tile.compilation.export_kernel", export)
    monkeypatch.setattr(_cutile, "validate_kernel_parameters", lambda *args: None)
    layout = _layout({"x": "fp32", "out": "fp32"})
    facts = _cutile.array_layout_facts("ns::op", layout, pointer_alignment=16)
    with pytest.raises(RuntimeError, match="tileiras"):
        _cutile.compile_cutile_to_ptx(
            "ns::op", object(), layout, {}, "sm_120", array_facts=facts
        )
    assert [array.base_addr_divisible_by for array in captured["arrays"]] == [1, 1]
    assert [array.stride_constant for array in captured["arrays"]] == [(1,), (1,)]


_RMSNORM_SIGNATURE = {"x": "bf16[2]", "w": "bf16[1]", "out": "bf16[2]"}
_RMSNORM_SHAPES = {"x": (256, 2048), "w": (2048,), "out": (256, 2048)}

try:
    import cuda.tile as ct

    # U4's RMSNorm: no allow_tma argument, so only the plugin build disables TMA.
    @ct.kernel
    def _rmsnorm(x, w, out, N: ct.Constant[int], EPS: ct.Constant[float]):
        row = ct.bid(0)
        xt = ct.astype(ct.load(x, index=(row, 0), shape=(1, N)), ct.float32)
        wt = ct.astype(ct.load(w, index=(0,), shape=(N,)), ct.float32)
        ms = ct.sum(xt * xt, axis=1, keepdims=True) / N
        y = xt * ct.rsqrt(ms + EPS) * ct.reshape(wt, (1, N))
        ct.store(out, index=(row, 0), tile=ct.astype(y, ct.bfloat16))

    # Pointer loads and stores, which tileiras rejects the allow_tma hint on.
    @ct.kernel
    def _gather_scatter(x, out, TILE: ct.Constant[int]):
        index = ct.bid(0) * TILE + ct.arange(TILE, dtype=ct.int32)
        ct.scatter(out, index, ct.gather(x, index) + 1.0)

except ImportError:
    _rmsnorm = _gather_scatter = None


def _rmsnorm_kernel():
    if _rmsnorm is None:
        pytest.skip("cuda-tile not installed")
    return _rmsnorm


@pytest.mark.parametrize(
    "aligned,vectorized", [(False, False), (True, True)], ids=["element", "aligned16"]
)
def test_aligned_static_cutile_export_vectorizes_without_tma(aligned, vectorized):
    """U4's RMSNorm: TMA would add descriptor parameters the AOT launch lacks."""
    reason = _cutile_toolchain_reason()
    if reason is not None:
        pytest.skip(reason)
    kernel = _rmsnorm_kernel()
    layout = _layout(_RMSNORM_SIGNATURE, names=("x", "w"))
    facts = _cutile.array_layout_facts(
        "ns::op",
        layout,
        pointer_alignment=16 if aligned else 1,
        static_shapes=_RMSNORM_SHAPES,
    )
    arch = (
        "sm_{}{}".format(*torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else "sm_120"
    )
    ptx, _, _ = _cutile.compile_cutile_to_ptx(
        "ns::op", kernel, layout, {"N": 2048, "EPS": 1e-6}, arch, array_facts=facts
    )
    _, _, params = _common.parse_entry(ptx.decode())
    assert len(params) == layout.num_slots
    ops = _mem_ops(ptx.decode())
    assert ("ld.global.v4.b32" in ops) is vectorized
    assert ("st.global.v4.b32" in ops) is vectorized


@skip_no_cuda
@skip_no_qdp
def test_aligned_static_cutile_op_runs_in_an_engine_and_declines_other_shapes():
    reason = _cutile_toolchain_reason()
    if reason is not None:
        pytest.skip(reason)
    import torch_tensorrt.kernels as ttk

    kernel = _rmsnorm_kernel()

    def meta(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def eager(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * rstd * w.float()).to(x.dtype)

    if not hasattr(torch.ops.frontend_fixes, "cutile_rmsnorm_aligned"):
        ttk.cutile_op(
            "frontend_fixes::cutile_rmsnorm_aligned",
            kernel=kernel,
            signature=_RMSNORM_SIGNATURE,
            meta_fn=meta,
            grid=lambda inputs, outputs: (inputs[0].shape_expr[0],),
            constants={"N": 2048, "EPS": 1e-6},
            eager_fn=eager,
            pointer_alignment=16,
            static_shapes=_RMSNORM_SHAPES,
            supports_dynamic_shapes=False,
        )

    class Model(torch.nn.Module):
        def forward(self, x, w):
            return torch.ops.frontend_fixes.cutile_rmsnorm_aligned(x, w)

    def compile_for(x, w):
        return torch_tensorrt.compile(
            Model().eval().cuda(), ir="dynamo", inputs=[x, w], min_block_size=1
        )

    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    compiled = compile_for(x, w)
    assert not any(
        "cutile_rmsnorm_aligned" in str(node.target) for node in compiled.graph.nodes
    ), "the op matching static_shapes was not lowered to its plugin"
    with torch.no_grad():
        torch.testing.assert_close(
            compiled(x, w), eager(x, w), atol=1.6e-2, rtol=1.6e-2
        )

    other = torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16)
    declined = compile_for(other, w)
    assert any(
        "cutile_rmsnorm_aligned" in str(node.target) for node in declined.graph.nodes
    ), "a shape other than static_shapes was lowered to the specialized plugin"
    with torch.no_grad():
        torch.testing.assert_close(declined(other, w), eager(other, w))


@skip_no_cuda
@skip_no_qdp
def test_aligned_cutile_op_with_gather_and_scatter_runs_in_an_engine():
    """The TMA-free build disabled TMA on ct.gather's and ct.scatter's pointer
    loads and stores too, which tileiras rejects."""
    reason = _cutile_toolchain_reason()
    if reason is not None:
        pytest.skip(reason)
    if _gather_scatter is None:
        pytest.skip("cuda-tile not installed")
    import tensorrt.plugin as trtp
    import torch_tensorrt.kernels as ttk

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    if not hasattr(torch.ops.frontend_fixes, "cutile_gather_aligned"):
        ttk.cutile_op(
            "frontend_fixes::cutile_gather_aligned",
            kernel=_gather_scatter,
            signature={"x": "fp32", "out": "fp32"},
            meta_fn=meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), 256),
            ),
            constants={"TILE": 256},
            pointer_alignment=16,
        )

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.frontend_fixes.cutile_gather_aligned(x)

    x = torch.randn(4096, device="cuda")
    compiled = torch_tensorrt.compile(
        Model().eval().cuda(), ir="dynamo", inputs=[x], min_block_size=1
    )
    assert not any(
        "cutile_gather_aligned" in str(node.target) for node in compiled.graph.nodes
    ), "the op was not lowered to its plugin"
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), x + 1, rtol=0, atol=0)


@skip_no_qdp
def test_cutile_op_requires_static_engines_for_static_shapes(monkeypatch):
    import torch_tensorrt.kernels as ttk

    monkeypatch.setattr(
        _cutile,
        "compile_cutile_to_ptx",
        lambda *args, **kwargs: pytest.fail("compiled an invalid registration"),
    )

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    with pytest.raises(ValueError, match="supports_dynamic_shapes=False"):
        ttk.cutile_op(
            "frontend_fixes::cutile_dynamic_static",
            kernel=object(),
            signature={"x": "fp32", "out": "fp32"},
            meta_fn=meta,
            grid=lambda inputs, outputs: 1,
            static_shapes={"x": (8,), "out": (8,)},
        )
