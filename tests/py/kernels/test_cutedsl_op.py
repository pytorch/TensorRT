# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CuTe DSL ABI checks and execution of serialized AOT kernels."""

import inspect
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tensorrt as trt
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels import _common, _cutedsl, _register

from .conftest import register_once, skip_no_cuda, skip_no_qdp

SIGNATURE = {"x": "*fp32", "n": "i32", "out": "*fp32"}
PTX = """.version 9.0
.target sm_90
.address_size 64
.visible .entry kernel(
    .param .u64 .ptr .global .align 1 x,
    .param .u32 n,
    .param .u64 .ptr .global .align 1 out
)
.reqntid 128, 1, 1
{ ret; }
"""


def _meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@pytest.fixture
def fake_kernel(monkeypatch):
    constexpr, dsl = object(), object()

    def kernel(x, n, out, block: constexpr):
        pass

    kernel._decorator_kind, kernel._dsl_cls = "kernel", dsl
    cutlass = SimpleNamespace(Constexpr=constexpr, Float32=object(), Int32=int)
    cute = SimpleNamespace(
        kernel=SimpleNamespace(__self__=dsl),
        jit=lambda fn: fn,
        AddressSpace=SimpleNamespace(gmem=object()),
    )
    monkeypatch.setattr(_cutedsl, "_cutedsl_import", lambda: (cutlass, cute))
    return kernel, cute


@skip_no_qdp
@pytest.mark.parametrize(
    "override,match",
    [
        ({"grid": None}, "grid must be callable"),
        ({"block_size": 1025}, "1024 threads"),
        ({"shared_mem": -1}, "non-negative"),
    ],
)
def test_invalid_launch_config_precedes_compilation(monkeypatch, override, match):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid registration reached compiler or registry")

    monkeypatch.setattr(_cutedsl, "compile_to_ptx", unexpected)
    monkeypatch.setattr(_register, "register_precompiled_qdp_plugin", unexpected)
    options = dict(
        op_name="cutedsl_tests::invalid",
        kernel=object(),
        signature={"x": "*fp32", "out": "*fp32"},
        meta_fn=_meta,
        grid=lambda inputs, outputs: 1,
        block_size=128,
    )
    options.update(override)
    with pytest.raises(ValueError, match=match):
        ttk.cutedsl_op(**options)


@pytest.mark.parametrize(
    "signature,constants,extras",
    [
        ({"out": "*fp32", "n": "i32", "x": "*fp32"}, {"block": 128}, lambda i, o: [1]),
        (SIGNATURE, {}, lambda i, o: [1]),
        (SIGNATURE, {"block": 128, "unknown": 1}, lambda i, o: [1]),
        (SIGNATURE, {"block": 128}, None),
    ],
)
def test_runtime_and_constexpr_parameters_must_match(
    fake_kernel, signature, constants, extras
):
    with pytest.raises(ValueError, match="runtime parameters|extra_args_fn"):
        _cutedsl.validate_config(
            "ns::op", fake_kernel[0], signature, constants, (1, 1), extras
        )


@pytest.mark.parametrize(
    "old,new,match",
    [
        (".param .u32 n", ".param .u64 n", "unsupported PTX ABI"),
        (".ptr .global .align 1", ".ptr .shared .align 1", "unsupported PTX ABI"),
        (".param .u32 n,", "", "exactly the declared arguments"),
        (".reqntid 128, 1, 1", ".reqntid 64, 1, 1", "thread dimensions"),
        ("{ ret; }", ".explicitcluster\n{ ret; }", "clustered"),
        ("{ ret; }", ".extern .shared .b8 work[];\n{ ret; }", "dynamic shared memory"),
    ],
)
def test_ptx_must_preserve_the_declared_abi_and_launch(old, new, match):
    layout = _common.analyze_signature(SIGNATURE, (1, 1))
    with pytest.raises((RuntimeError, ValueError), match=match):
        _cutedsl._validate_ptx(PTX.replace(old, new), layout, (128, 1, 1), 0)


def test_compiler_added_workspace_is_rejected(fake_kernel, monkeypatch):
    runtime = ModuleType("cutlass.cute.runtime")
    runtime.make_ptr = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "cutlass.cute.runtime", runtime)
    compiler = ModuleType("cutlass.base_dsl.compiler")
    for name in ("GPUArch", "KeepPTX", "DumpDir", "ExtraCompilerOpts"):
        setattr(compiler, name, lambda *args: object())
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.compiler", compiler)
    fake_kernel[1].compile = MagicMock()
    fake_kernel[1].compile.__getitem__.return_value.return_value = SimpleNamespace(
        __ptx__=PTX, total_added_arguments=1, kernel_extra_args={}
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(_cutedsl, "_target_arch", lambda arch: "sm_90a")
    monkeypatch.setattr(_cutedsl, "_driver_ptx_version", lambda: 90)
    with pytest.raises(RuntimeError, match="workspace"):
        _cutedsl.compile_to_ptx(
            fake_kernel[0],
            {"block": 128},
            _common.analyze_signature(SIGNATURE, (1, 1)),
            (128, 1, 1),
            0,
        )


@pytest.mark.parametrize(
    "arch,target",
    [
        ((8, 0), "sm_80"),
        ((8, 9), "sm_89"),
        ((9, 0), "sm_90a"),
        ((10, 0), "sm_100a"),
        ((10, 1), "sm_110a"),
        ((12, 0), "sm_120a"),
    ],
)
def test_target_preserves_architecture_features(arch, target, monkeypatch):
    pytest.importorskip("cutlass.base_dsl")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_80")
    assert _cutedsl._target_arch(arch) == target


def test_unsupported_target_fails_clearly():
    pytest.importorskip("cutlass.base_dsl")
    with pytest.raises(RuntimeError, match="does not support CUDA target sm_70"):
        _cutedsl._target_arch((7, 0))


@pytest.mark.parametrize(
    "version,expected",
    [(12050, 85), (12060, 85), (12080, 87), (12090, 88), (13000, 90), (13030, 93)],
)
def test_driver_ptx_ceiling_uses_loaded_driver(version, expected, monkeypatch):
    driver = pytest.importorskip("cuda.bindings.driver")
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: (0, version))
    assert _cutedsl._driver_ptx_version() == expected


@pytest.mark.parametrize("result", [(0, 0), (0, 11080), (1, 13030)])
def test_unavailable_driver_ptx_support_fails_clearly(result, monkeypatch):
    driver = pytest.importorskip("cuda.bindings.driver")
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: result)
    with pytest.raises(RuntimeError, match="queryable PTX support"):
        _cutedsl._driver_ptx_version()


@pytest.mark.parametrize("emitted", [88, 93, 94])
def test_driver_isa_ceiling_recompiles_only_newer_ptx(emitted, monkeypatch):
    monkeypatch.setattr(_cutedsl, "_driver_ptx_version", lambda: 93)
    calls = []

    def compile(version):
        calls.append(version)
        isa = emitted if version is None else version
        # A distinct body proves the compiler's new output is used, rather than
        # only relabeling the original PTX header.
        return f".version {isa // 10}.{isa % 10}\n// compiled for {version}\n"

    ptx = _cutedsl._compile_within_driver_isa(compile)
    if emitted > 93:
        assert calls == [None, 93]
        assert ptx == ".version 9.3\n// compiled for 93\n"
    else:
        assert calls == [None]
        assert ptx == f".version {emitted // 10}.{emitted % 10}\n// compiled for None\n"


def test_compiler_must_honor_driver_isa_ceiling(monkeypatch):
    monkeypatch.setattr(_cutedsl, "_driver_ptx_version", lambda: 87)
    with pytest.raises(RuntimeError, match="did not honor.*8.7"):
        _cutedsl._compile_within_driver_isa(lambda version: PTX)


def test_unsupported_compiler_isa_has_actionable_error(monkeypatch):
    monkeypatch.setattr(_cutedsl, "_driver_ptx_version", lambda: 87)
    failure = ValueError("unsupported compiler target")

    def compile(version):
        if version is not None:
            raise failure
        return PTX

    with pytest.raises(RuntimeError, match="PTX ISA 8.7.*Upgrade") as error:
        _cutedsl._compile_within_driver_isa(compile)
    assert error.value.__cause__ is failure


@skip_no_cuda
def test_hopper_warpgroup_instruction_compiles(tmp_path):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    from cutlass.base_dsl.compiler import DumpDir, GPUArch, KeepPTX
    from cutlass.cute.runtime import make_ptr

    @cute.kernel
    def kernel(x: cute.Pointer, out: cute.Pointer):
        cute.nvgpu.warpgroup.fence()
        out[0] = x[0]

    @cute.jit
    def launch(x: cute.Pointer, out: cute.Pointer):
        kernel(x, out).launch(grid=(1, 1, 1), block=(128, 1, 1))

    ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=1)
    # Compile only: this verifies Hopper code generation even on another GPU.
    compiled = cute.compile[
        GPUArch(_cutedsl._target_arch((9, 0))), KeepPTX(), DumpDir(str(tmp_path))
    ](launch, ptr, ptr)
    assert ".target sm_90a" in compiled.__ptx__
    assert "wgmma.fence.sync.aligned" in compiled.__ptx__


def _run_plan(plan, x, y):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    layers = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    assert "cutedsl_tests.sum_difference_" in layers
    context = engine.create_execution_context()
    inputs, outputs = [], []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        (
            inputs
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else outputs
        ).append(name)
    assert len(inputs) == len(outputs) == 2
    for name, tensor in zip(inputs, (x, y)):
        assert context.set_input_shape(name, tuple(tensor.shape))
        assert context.set_tensor_address(name, tensor.data_ptr())
    result = [
        torch.empty(
            tuple(context.get_tensor_shape(name)), device=x.device, dtype=x.dtype
        )
        for name in outputs
    ]
    for name, tensor in zip(outputs, result):
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return result


@skip_no_cuda
@skip_no_qdp
@pytest.mark.parametrize(
    "dtype,ptx_ceiling",
    [(torch.float16, None), (torch.float32, None), (torch.float32, 87)],
    ids=["fp16", "fp32", "fp32-driver-ptx87"],
)
def test_dynamic_multi_output_engine_and_standalone_serialization(
    dtype, ptx_ceiling, tmp_path, monkeypatch
):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    import tensorrt.plugin as trtp
    from cutlass.memory import SmemAllocator

    if ptx_ceiling is not None:
        if torch.cuda.get_device_capability() not in {
            (8, 0),
            (8, 6),
            (8, 7),
            (8, 9),
            (9, 0),
            (10, 0),
            (12, 0),
        }:
            pytest.skip("This forced PTX 8.7 case requires a GPU supported by PTX 8.7")
        ceiling = min(ptx_ceiling, _cutedsl._driver_ptx_version())
        monkeypatch.setattr(_cutedsl, "_driver_ptx_version", lambda: ceiling)
        load_ptx = _common.load_ptx

        def checked_load(ptx, name):
            assert _cutedsl._ptx_version(ptx) <= ceiling
            return load_ptx(ptx, name)

        monkeypatch.setattr(_common, "load_ptx", checked_load)

    @cute.kernel
    def kernel(
        x: cute.Pointer,
        y: cute.Pointer,
        n: cutlass.Int32,
        summed: cute.Pointer,
        difference: cute.Pointer,
        block: cutlass.Constexpr,
        use_shared: cutlass.Constexpr,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        index = bx * block + tx
        if cutlass.const_expr(use_shared):
            buffer = cute.recast_ptr(
                SmemAllocator().allocate(block * 4, byte_alignment=4),
                dtype=cutlass.Float32,
            )
            if index < n:
                buffer[tx] = x[index] + y[index]
            cute.arch.barrier()
            source = block - 1 - tx
            target = bx * block + source
            if target < n:
                summed[target] = buffer[source]
        else:
            if index < n:
                summed[index] = x[index] + y[index]
        if index < n:
            difference[index] = x[index] - y[index]

    def meta(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(x), torch.empty_like(x)

    def eager(x, y):
        return x + y, x - y

    spelling = "fp16" if dtype == torch.float16 else "fp32"
    name = f"sum_difference_{spelling}_{ptx_ceiling}"
    op_name = f"cutedsl_tests::{name}"
    signature = {
        "x": f"*{spelling}",
        "y": f"*{spelling}",
        "n": "i32",
        "summed": f"*{spelling}",
        "difference": f"*{spelling}",
    }
    register_once(
        op_name,
        lambda: ttk.cutedsl_op(
            op_name,
            kernel,
            signature,
            meta,
            grid=lambda i, o: trtp.cdiv(i[0].shape_expr.numel(), 128),
            block_size=128,
            constants={"block": 128, "use_shared": dtype == torch.float32},
            shared_mem=512 if dtype == torch.float32 else 0,
            eager_fn=eager,
            extra_args_fn=lambda i, o: [trtp.SymInt32(i[0].shape_expr.numel())],
        ),
    )
    target = getattr(torch.ops.cutedsl_tests, name)

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return target(x, y)

    example = tuple(torch.randn(257, device="cuda", dtype=dtype) for _ in range(2))
    dim = torch.export.Dim("elements", min=31, max=1025)
    exported = torch.export.export(
        Model(), example, dynamic_shapes=({0: dim}, {0: dim})
    )
    profiles = [
        torch_tensorrt.Input(
            min_shape=(31,), opt_shape=(257,), max_shape=(1025,), dtype=dtype
        )
        for _ in range(2)
    ]
    plan = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exported,
        inputs=profiles,
        enabled_precisions={dtype},
    )
    assert plan
    for size in (31, 129, 1025):
        x, y = (torch.randn(size, device="cuda", dtype=dtype) for _ in range(2))
        for actual, expected in zip(_run_plan(plan, x, y), eager(x, y)):
            torch.testing.assert_close(actual, expected)

    if dtype == torch.float32:
        path = tmp_path / "cutedsl.plan"
        path.write_bytes(plan)
        child = (
            """
import importlib.abc
import sys
import torch
import tensorrt as trt
torch.cuda.init()
for name in list(sys.modules):
    if name == "cutlass" or name.startswith("cutlass."):
        del sys.modules[name]
class BlockCute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass" or fullname.startswith("cutlass."):
            raise ImportError("CuTe DSL intentionally unavailable")
sys.meta_path.insert(0, BlockCute())
"""
            + inspect.getsource(_run_plan)
            + """
x, y = (torch.randn(513, device="cuda") for _ in range(2))
for actual, expected in zip(_run_plan(open(sys.argv[1], "rb").read(), x, y), (x + y, x - y)):
    torch.testing.assert_close(actual, expected)
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
"""
        )
        result = subprocess.run(
            [sys.executable, "-c", child, str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
