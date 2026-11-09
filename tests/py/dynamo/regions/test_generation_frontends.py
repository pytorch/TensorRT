# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The builtin CuTe DSL backend, backend options, and front-end registrations.

CPU cases need only the Python packages. GPU cases skip without CUDA, a
QDP-capable TensorRT, or a front end's toolchain that supports this GPU and
driver; with that toolchain present, a backend that cannot compile and load its
probe kernel fails them (for CuTe DSL, only with
TORCH_TENSORRT_STRICT_TOOLCHAIN_TESTS=1: see require_cutedsl). Worker cases
spawn the real evaluation worker exactly as the generation controller does.
"""

import copy
import inspect
import json
import os
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import torch_tensorrt
from packaging.version import Version
from torch_tensorrt import kernels
from torch_tensorrt.dynamo.region_compilation._generation import _candidate
from torch_tensorrt.dynamo.region_compilation._generation._backends import (
    get_backend,
    validate_tuning_space,
)
from torch_tensorrt.dynamo.region_compilation._generation._instructions import (
    load_instructions,
)
from torch_tensorrt.dynamo.region_compilation._plugins import _TensorBoundary
from torch_tensorrt.region import KernelGenerator, SearchPolicy

from ._generation_utils import require_cuda_cpp_toolchain

requires_gpu_plugins = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="requires CUDA and the TensorRT QDP plugin",
)

CUTE_ADD = """import cutlass
import cutlass.cute as cute

@cute.kernel
def add(arg0, arg1, out0, N: cutlass.Constexpr, BLOCK: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * BLOCK + tidx
    if i < N:
        out0[i] = arg0[i] + arg1[i]
"""


def cute_add(**updates):
    return {
        "source": CUTE_ADD,
        "kernel_name": "add",
        "grid": [4],
        "block": [256],
        "shared_mem_bytes": 0,
        "constants": [{"name": "N", "value": 1000}, {"name": "BLOCK", "value": 256}],
    } | updates


def with_line(line):
    anchor = "    tidx, _, _"
    return cute_add(source=CUTE_ADD.replace(anchor, f"    {line}\n{anchor}", 1))


# Driver results for the probe's PTX that every CuTe DSL kernel would get: the
# driver's PTX ISA is older than the DSL's, or it cannot JIT PTX at all.
_DRIVER_REJECTIONS = (
    "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
    "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
    "CUDA_ERROR_JIT_COMPILATION_DISABLED",
)


# CI's regions-cutedsl suite sets this to "1" through Suite.env
# (tests/ci/suites.py), as it installs a pinned, tested nvidia-cutlass-dsl.
STRICT_TOOLCHAIN_TESTS = "TORCH_TENSORRT_STRICT_TOOLCHAIN_TESTS"


def require_cutedsl():
    """Skip unless this host's CuTe DSL toolchain supports generation.

    By default any failure of the backend's toolchain check skips, with its
    reason: dynamo-regions gets nvidia-cutlass-dsl unpinned, through
    flashinfer-python, and a new DSL release must not fail unrelated changes.
    With TORCH_TENSORRT_STRICT_TOOLCHAIN_TESTS=1, which regions-cutedsl sets
    after installing a pinned, tested release, only a missing or unsupported
    toolchain skips: the DSL or cuda-core is absent; the DSL is outside the
    supported range, cannot target this GPU, or is a release the frontend
    reports it is not tested with; or the driver cannot load the PTX ISA the
    DSL compiles to. Any other failure to compile and load the backend's probe
    kernel then fails the test, so that a broken backend cannot pass as a
    missing toolchain.
    """
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cuda.core")
    try:
        get_backend("cutedsl").target_metadata("cuda")
    except Exception as error:
        strict = os.environ.get(STRICT_TOOLCHAIN_TESTS) == "1"
        if not strict or _unsupported_toolchain(error):
            pytest.skip(str(error))
        raise


def _unsupported_toolchain(error):
    if str(error).startswith(
        ("CuTe DSL generation is unavailable", "CuTe DSL generation cannot target")
    ):
        return True
    if "this frontend is tested with" in str(error):
        # The frontend's report of API drift. On a release it is tested with,
        # the report itself is the bug: validate_config cannot tell a kernel
        # whose decorator internals changed from an object that is no kernel.
        return not _tested_release()
    # The probe kernel failed: only the driver's PTX ISA ceiling, which no
    # kernel avoids, makes that a limit of the toolchain rather than a bug.
    cause = error.__cause__
    return isinstance(cause, RuntimeError) and (
        "driver PTX ISA" in str(cause)
        or any(code in str(cause) for code in _DRIVER_REJECTIONS)
    )


def _tested_release():
    """Whether the installed DSL is in the series its frontend is tested with."""
    import cutlass

    from torch_tensorrt.kernels import _cutedsl as frontend

    series = Version(frontend._TESTED_SERIES).release
    return Version(cutlass.__version__).release[: len(series)] == series


# ---- CPU: contract, names and source rules ----


def test_cutedsl_is_a_builtin_with_a_structured_output_schema():
    backend = get_backend("cutedsl")
    assert backend is get_backend("cutedsl")
    assert backend.instruction_resources == ("references/cutedsl.md",)
    schema = backend.proposal_schema
    assert set(schema["properties"]) == set(cute_add())
    assert schema["properties"]["block"]["items"]["maximum"] == 1024
    assert schema["properties"]["shared_mem_bytes"]["maximum"] == 49152
    allowed = {
        "type",
        "enum",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "items",
        "anyOf",
        "properties",
        "required",
        "additionalProperties",
    }

    def keywords(node):
        assert set(node) <= allowed, set(node) - allowed
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
        for child in node.get("properties", {}).values():
            keywords(child)
        if "items" in node:
            keywords(node["items"])

    keywords(schema)


def test_unknown_backend_error_lists_cutedsl():
    with pytest.raises(ValueError, match="'cutedsl'"):
        get_backend("cuda_python")


def test_kernel_generator_accepts_cutedsl_and_checks_its_tuning():
    tuning = {"block": ([128], [256]), "constants.BLOCK": (128, 256)}
    assert KernelGenerator(backend="cutedsl", search=SearchPolicy(tuning=tuning))
    assert validate_tuning_space("cutedsl", tuning)["block"] == [[128], [256]]
    # Alignment is the user's aligned_io promise, not an author or tuning choice.
    with pytest.raises(ValueError, match="no tunable field 'aligned16'"):
        KernelGenerator(
            backend="cutedsl", search=SearchPolicy(tuning={"aligned16": (True,)})
        )


def test_instructions_select_the_cutedsl_guide():
    backend = get_backend("cutedsl")
    bundle = load_instructions("cutedsl", backend.instruction_resources)
    resources = {item["name"]: item["text"] for item in bundle["resources"]}
    assert "references/cutedsl.md" in resources
    assert "references/cutedsl.md" in resources["SKILL.md"]
    assert "codegen.aligned_io" in resources["references/cutedsl.md"]


def test_instructions_cli_prints_the_cutedsl_guide_and_schema():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_tensorrt.dynamo.region_compilation._generation",
            "instructions",
            "--backend",
            "cutedsl",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "INSTRUCTION RESOURCE: references/cutedsl.md" in result.stdout
    schema = json.loads(result.stdout.split("PROPOSAL SCHEMA:\n", 1)[1])
    assert schema == get_backend("cutedsl").proposal_schema


def test_cutedsl_accepts_the_guide_example():
    backend = get_backend("cutedsl")
    assert backend.validate_candidate(cute_add(), 2, 1)["block"] == [256]


def test_cutedsl_accepts_realistic_device_code():
    # Methods of DSL objects, such as SmemAllocator.allocate_tensor, are known
    # from the installed CuTe DSL.
    pytest.importorskip("cutlass.cute")
    backend = get_backend("cutedsl")
    realistic = with_line(
        "smem = cutlass.utils.SmemAllocator()\n"
        "    buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BLOCK), byte_alignment=16)\n"
        "    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)))\n"
        "    st = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4), cutlass.Float16)\n"
        "    cp = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32)\n"
        "    for k in cutlass.range_constexpr(4):\n        cute.arch.barrier()\n"
        "    s = cute.arch.warp_reduction_sum(cutlass.Float32(1.0))\n"
        "    o = cute.assume(4, divby=4)"
    )
    backend.validate_candidate(realistic, 2, 1)


@pytest.mark.parametrize(
    "proposal,match",
    [
        (
            cute_add(source=CUTE_ADD.replace("@cute.kernel", "@cute.jit")),
            "@cute.kernel",
        ),
        (
            cute_add(
                source=CUTE_ADD.replace("def add(arg0,", "def add(arg0: cute.Pointer,")
            ),
            "annotations",
        ),
        (with_line("cute.nvgpu.warpgroup.fence()"), "warpgroup"),
        (with_line("a = cute.nvgpu.make_tiled_tma_atom_A(None)"), "tma"),
        (with_line("cute.nvgpu.tcgen05.fence()"), "tcgen05"),
        (with_line("cute.arch.alloc_tmem(32, None)"), "alloc_tmem"),
        (with_line("cute.printf(1)"), "printf"),
        (with_line("p = cute.make_ptr(cutlass.Float32, 0)"), "make_ptr"),
        (with_line("a = arg0.align(16)"), "arg0.align"),
        (with_line("cutlass.cute.compile(None)"), "compile"),
        (with_line("cutlass.torch.empty(1)"), "torch"),
        (with_line("cute.runtime.from_dlpack(None)"), "runtime"),
        (
            cute_add(
                source=CUTE_ADD.replace(
                    "BLOCK: cutlass.Constexpr", "cute: cutlass.Constexpr"
                ),
                constants=[
                    {"name": "N", "value": 1000},
                    {"name": "cute", "value": 256},
                ],
            ),
            "nonreserved",
        ),
        (cute_add(block=[2048]), "1024 threads"),
        (cute_add(block=[1, 1, 65]), "z dimension"),
        (cute_add(shared_mem_bytes=65536), "49152"),
        (cute_add(aligned16=False), "must contain exactly"),
    ],
)
def test_cutedsl_rejects_unsupported_source_and_launches(proposal, match):
    with pytest.raises(ValueError, match=match):
        get_backend("cutedsl").validate_candidate(proposal, 2, 1)


@pytest.mark.parametrize(
    "line,match",
    [
        # This kernel passed the source check, then ran a host command while
        # CuTe DSL traced it, and still computed the right output.
        ("abs = cutlass.cutlass_dsl.os.system\n    abs('true')", "cannot bind 'abs'"),
        (
            "system = cutlass.cutlass_dsl.os.system",
            r"'cutlass\.cutlass_dsl\.os\.system' is not allowed",
        ),
        # cute.transform_leaf calls its first argument while tracing.
        ("cute.transform_leaf(open, 'marker', 'w')", "'open' is not a parameter"),
        ("cute.transform_leaf(eval, '1')", "'eval' is not a parameter"),
        # CuTe DSL moves loop and branch bodies into nested functions, so open
        # here was the builtin, which created the file while tracing.
        (
            "for j in range(N):\n        open = arg0\n"
            "    cute.transform_leaf(open, 'marker', 'w')",
            "'open' names a Python builtin wherever",
        ),
        (
            "if N > 1:\n        open = arg0\n    cute.transform_leaf(open, 'marker', 'w')",
            "'open' names a Python builtin wherever",
        ),
        # A comprehension's variables do not outlive it.
        (
            "first = [0 for eval in (1,)]\n    cute.transform_leaf(eval, '1')",
            "'eval' names a Python builtin wherever",
        ),
        # The denials hold for every use of a name, not only for calls.
        ("convert = cute.runtime.from_dlpack", "runtime"),
        ("check = cute.testing", "testing"),
        ("modules = (cute,)", "DSL module 'cute' only"),
        ("namespace = cutlass.cute", "not modules or other objects"),
        ("cute.arch.thread_idx = cute.arch.lane_idx", "cannot assign or delete"),
    ],
)
def test_cutedsl_kernel_code_stays_within_the_dsl(line, match):
    with pytest.raises(ValueError, match=match):
        get_backend("cutedsl").validate_candidate(with_line(line), 2, 1)


# CuTe DSL's rewrite of a kernel calls type() for `and` and `or`, and locals()
# for the variables of loops and branches: a local of either name made those
# kernels fail to compile, with "'Int32' object is not callable".
@pytest.mark.parametrize("name", ["type", "locals"])
def test_cutedsl_kernel_code_cannot_rebind_builtins_its_compiler_calls(name):
    with pytest.raises(ValueError, match=f"cannot bind '{name}'"):
        get_backend("cutedsl").validate_candidate(with_line(f"{name} = arg0"), 2, 1)
    proposal = with_line("sum = arg0\n    id = sum")
    assert get_backend("cutedsl").validate_candidate(proposal, 2, 1)


@pytest.mark.parametrize(
    "line,match",
    [
        # Modules that cute.math and cute.arch import: as values they reached
        # functools.reduce and os.rename, which renamed a file while tracing.
        (
            "moved = (cute.math.functools,)[0].reduce("
            "(cute.arch.cutlass_dsl,)[0].os.rename, ('dst',), 'src')",
            "not modules or other objects",
        ),
        ("namespace = cute.nvgpu.warp", "not modules or other objects"),
        # A partial object, not a class, so its attributes are not DSL names.
        ("shuffle = cute.arch.shuffle_sync_bfly.func", "not modules or other objects"),
        # What the namespaces import from the standard library is not CuTe's.
        (
            "cls = cute.nvgpu.cpasync.dataclass(cutlass.Float32)",
            r"'cute\.nvgpu\.cpasync\.dataclass' is not allowed",
        ),
        ("meta = cute.nvgpu.cpasync.ABCMeta", "not modules or other objects"),
        (
            "lane = cute.arch.partial(cute.arch.lane_idx)",
            r"'cute\.arch\.partial' is not allowed",
        ),
    ],
)
def test_cutedsl_rejects_modules_and_objects_inside_its_namespaces(line, match):
    pytest.importorskip("cutlass.cute")
    with pytest.raises(ValueError, match=match):
        get_backend("cutedsl").validate_candidate(with_line(line), 2, 1)


def test_cutedsl_accepts_enum_members_and_numeric_type_attributes():
    proposal = with_line(
        "op = cute.ReductionOp.ADD\n"
        "    mode = cute.nvgpu.cpasync.LoadCacheMode.GLOBAL\n"
        "    width = cutlass.Float32.width\n"
        "    lane = cutlass.cute.arch.lane_idx()"
    )
    backend = get_backend("cutedsl")
    assert backend.validate_candidate(proposal, 2, 1)["source"] == proposal["source"]


def test_cutedsl_aliases_do_not_reserve_names_for_other_backends():
    assert _candidate._identifier("cute") and _candidate._identifier("cutlass")
    assert not _candidate._identifier("tl")
    assert not _candidate._identifier("cute", get_backend("cutedsl").aliases)


@pytest.mark.parametrize(
    "options,expected",
    [(None, False), ({}, False), ({"aligned_io": True}, True), ({"other": 1}, False)],
)
def test_aligned_io_option(options, expected):
    assert _candidate._aligned_io(options) is expected


def test_aligned_io_option_must_be_a_bool():
    with pytest.raises(ValueError, match="aligned_io must be a bool"):
        _candidate._aligned_io({"aligned_io": 1})


@pytest.mark.parametrize("name", ["cutile", "cuda_cpp", "cutedsl"])
def test_frontend_backends_accept_registration_options(name):
    parameters = inspect.signature(get_backend(name).make_registration).parameters
    assert parameters["options"].default is None


CUTILE_ADD = """import cuda.tile as ct

@ct.kernel
def add(arg0, out0, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    x = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,))
    ct.store(out0, index=(ct.bid(0),), tile=x + x)
"""


@pytest.mark.parametrize("aligned", [False, True])
def test_cutile_registration_carries_static_shapes_and_alignment(tmp_path, aligned):
    pytest.importorskip("cuda.tile")
    spec = _TensorBoundary((8, 32), torch.float32, torch.device("cuda:0"))
    proposal = {
        "source": CUTILE_ADD,
        "kernel_name": "add",
        "grid": [2],
        "constants": [{"name": "N", "value": 256}, {"name": "BLOCK", "value": 128}],
    }
    registration = get_backend("cutile").make_registration(
        proposal, [spec], [spec], tmp_path, options={"aligned_io": aligned}
    )
    assert isinstance(registration, partial) and registration.func is kernels.cutile_op
    assert "signature" not in registration.keywords
    assert registration.keywords["static_shapes"] == {"arg0": (8, 32), "out0": (8, 32)}
    assert registration.keywords["pointer_alignment"] == (16 if aligned else None)
    assert registration.keywords["ndim"] == 1


TRITON_ADD = """import triton
import triton.language as tl

@triton.jit
def add(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(arg0 + offsets, mask=offsets < N)
    tl.store(out0 + offsets, x + x, mask=offsets < N)
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("aligned", [False, True])
def test_triton_registration_compiles_and_carries_the_alignment(
    tmp_path, monkeypatch, aligned
):
    pytest.importorskip("triton")
    from torch_tensorrt.kernels import _triton

    alignments = []
    compile_triton = _triton._compile_triton

    def recording(*args, **kwargs):
        alignments.append(kwargs["pointer_alignment"])
        return compile_triton(*args, **kwargs)

    monkeypatch.setattr(_triton, "_compile_triton", recording)
    spec = _TensorBoundary((8, 32), torch.float32, torch.device("cuda:0"))
    proposal = {
        "source": TRITON_ADD,
        "kernel_name": "add",
        "grid": [2],
        "constants": [{"name": "N", "value": 256}, {"name": "BLOCK", "value": 128}],
        "num_warps": 4,
        "num_stages": None,
    }
    registration = get_backend("triton").make_registration(
        proposal, [spec], [spec], tmp_path, options={"aligned_io": aligned}
    )
    assert isinstance(registration, partial) and registration.func is kernels.triton_op
    expected = 16 if aligned else None
    assert registration.keywords["pointer_alignment"] == expected
    # The shared-memory check compiles the PTX the registration embeds.
    assert alignments == [expected]


# ---- GPU: registration, launch and the real worker ----


def _boundaries(*tensors):
    return [_TensorBoundary(tuple(t.shape), t.dtype, t.device) for t in tensors]


@requires_gpu_plugins
def test_cutedsl_target_metadata_checks_the_toolchain():
    require_cutedsl()
    metadata = get_backend("cutedsl").target_metadata("cuda")
    versions = metadata["versions"]
    assert versions["cute_target"].startswith("sm_")
    assert versions["driver_ptx"] is None or versions["driver_ptx"] >= 80
    assert "nvidia-cutlass-dsl" in versions


def _probe_load_error(code):
    return RuntimeError(
        "The CUDA driver could not load CuTe DSL PTX for 'torch_tensorrt_probe': "
        f"<CUresult.{code}: 0>. Use a CUDA driver and CuTe DSL toolchain that "
        "support this kernel's PTX ISA and target architecture."
    )


def _api_drift_report(kind):
    """The frontend's report of API drift, naming the series it is tested with."""
    import cutlass

    from torch_tensorrt.kernels import _cutedsl as frontend

    if kind == "metadata":
        return RuntimeError(frontend._api_changed(cutlass, "compiled-kernel metadata"))
    # validate_config gives the same report for an object that is no kernel.
    signature = {"arg0": "*fp32", "out0": "*fp32"}
    with pytest.raises(ValueError, match="this frontend is tested with") as report:
        frontend.validate_config("probe", len, signature, {}, (1, 1), None)
    return report.value


def _outcome(check):
    """The skip or the error that ``check`` raised; a skip is a result here."""
    try:
        check()
    except (pytest.skip.Exception, Exception) as outcome:
        return outcome
    raise AssertionError(f"{check.__name__} raised nothing")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
@pytest.mark.parametrize(
    "stage, error, skips",
    [
        ("import", ImportError("cutedsl_op requires nvidia-cutlass-dsl"), True),
        ("target", RuntimeError("Installed CuTe DSL does not support sm_99."), True),
        ("probe", RuntimeError("CuTe DSL did not honor the driver PTX ISA 8.7."), True),
        ("probe", _probe_load_error("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"), True),
        # The frontend's report of API drift in a release it is not tested
        # with. On a release it is tested with, that report is a backend bug.
        ("probe", ("metadata", "untested"), True),
        ("probe", ("decorator", "untested"), True),
        ("probe", ("metadata", "tested"), False),
        ("probe", ("decorator", "tested"), False),
        ("probe", _probe_load_error("CUDA_ERROR_INVALID_PTX"), False),
        ("probe", TypeError("a regression in the compile path"), False),
        ("target", AttributeError("type object 'Arch' has no attribute 'X'"), False),
    ],
    ids=[
        "no-dsl",
        "gpu",
        "driver-isa",
        "driver-ptx",
        "untested-metadata",
        "untested-decorator",
        "tested-metadata",
        "tested-decorator",
        "invalid-ptx",
        "backend-bug",
        "target-api",
    ],
)
def test_cutedsl_gpu_cases_fail_on_a_broken_backend_only_when_strict(
    monkeypatch, strict, stage, error, skips
):
    # regions-cutedsl runs the GPU cases with a pinned DSL: a broken backend
    # must fail them there rather than skip them as if the toolchain were
    # missing. Elsewhere no failure of an unpinned DSL may fail them.
    cutlass = pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cuda.core")
    from torch_tensorrt.dynamo.region_compilation._generation import _cutedsl
    from torch_tensorrt.kernels import _cutedsl as frontend

    # Each case fails one stage of the check and runs the others against the
    # installed DSL. Where that DSL already fails a stage, as a release the
    # frontend cannot use may, skip like the GPU cases these cases describe.
    monkeypatch.delenv(STRICT_TOOLCHAIN_TESTS, raising=False)
    require_cutedsl()
    if strict:
        monkeypatch.setenv(STRICT_TOOLCHAIN_TESTS, "1")
    report, release = error if isinstance(error, tuple) else (None, "tested")
    # A release of the series the frontend is tested with, or of the next one.
    major, minor = Version(frontend._TESTED_SERIES).release
    if release == "untested":
        minor += 1
    monkeypatch.setattr(cutlass, "__version__", f"{major}.{minor}.1")
    if report is not None:
        error = _api_drift_report(report)

    def fail(*args):
        raise error

    owner, name = {
        "import": (frontend, "_cutedsl_import"),
        "target": (frontend, "_target_arch"),
        "probe": (get_backend("cutedsl"), "_artifact"),
    }[stage]
    monkeypatch.setattr(owner, name, fail)
    # Probe again, even where an earlier test compiled the probe.
    monkeypatch.setattr(_cutedsl, "_TOOLCHAIN", set())
    outcome = _outcome(require_cutedsl)
    if strict and not skips:
        assert not isinstance(outcome, pytest.skip.Exception), str(outcome)
        assert error in (outcome, outcome.__cause__)
    else:
        assert isinstance(outcome, pytest.skip.Exception), repr(outcome)
        assert str(error) in str(outcome)


@pytest.mark.parametrize(
    "name, strict", [("regions-cutedsl", True), ("dynamo-regions", False)]
)
def test_cutedsl_gpu_cases_are_strict_only_in_the_suite_that_pins_the_dsl(
    monkeypatch, tmp_path, name, strict
):
    # dynamo-regions gets nvidia-cutlass-dsl through flashinfer-python, at
    # whatever release pip resolves, so a new one must not fail its cases.
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root))
    from tests.ci import runner

    monkeypatch.delenv(STRICT_TOOLCHAIN_TESTS, raising=False)
    monkeypatch.setenv("RUNNER_TEST_RESULTS_DIR", str(tmp_path))
    commands = []

    def record(argv, *, cwd, env):
        commands.append((argv, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=record))
    assert runner.run_suite(runner.by_name(name), "standard") == 0
    (environment,) = [env for argv, env in commands if "pytest" in argv]
    pinned = any(
        arg.startswith("nvidia-cutlass-dsl==") for argv, _ in commands for arg in argv
    )
    assert pinned is strict
    assert (environment.get(STRICT_TOOLCHAIN_TESTS) == "1") is strict


@requires_gpu_plugins
@pytest.mark.parametrize("aligned,align", [(False, "4"), (True, "16")])
def test_cutedsl_registration_honours_aligned_io(tmp_path, aligned, align):
    require_cutedsl()
    backend = get_backend("cutedsl")
    a, b = (torch.randn(1000, device="cuda") for _ in range(2))
    registration = backend.make_registration(
        cute_add(), _boundaries(a, b), _boundaries(a), tmp_path, {"aligned_io": aligned}
    )
    assert isinstance(registration, partial) and registration.func is kernels.ptx_op
    ptx = registration.keywords["ptx"].decode()
    assert set(re.findall(r"\.ptr\s+\.global\s+\.align\s+(\d+)", ptx)) == {align}
    # The screening launch runs the PTX this registration embeds.
    out = torch.empty_like(a)
    candidate = backend.validate_candidate(cute_add(), 2, 1)
    backend.launch(candidate, [a, b], [out], tmp_path)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)


@requires_gpu_plugins
def test_cutedsl_rejects_shared_memory_declared_as_zero(tmp_path):
    require_cutedsl()
    a = torch.randn(1000, device="cuda")
    # The sum round-trips through shared memory, so the allocation is not elided.
    source = CUTE_ADD.replace(
        "    tidx, _, _",
        "    smem = cutlass.utils.SmemAllocator()\n"
        "    buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BLOCK), byte_alignment=16)\n"
        "    tidx, _, _",
        1,
    ).replace(
        "        out0[i] = arg0[i] + arg1[i]",
        "        buf[tidx] = arg0[i] + arg1[i]\n        out0[i] = buf[tidx]",
    )
    assert "out0[i] = buf[tidx]" in source
    with pytest.raises(ValueError, match="dynamic shared memory"):
        get_backend("cutedsl").make_registration(
            cute_add(source=source), _boundaries(a, a), _boundaries(a), tmp_path
        )


def _evaluate(tmp_path, backend_name, fn, inputs, candidate, phase):
    from torch.fx.experimental.proxy_tensor import make_fx

    from torch_tensorrt.dynamo.region_compilation._generation._factory import (
        _compile_options,
        _describe,
        _reference,
    )
    from torch_tensorrt.dynamo.region_compilation._generation._reference import (
        save_reference,
    )
    from torch_tensorrt.dynamo.region_compilation._generation._util import digest

    backend = get_backend(backend_name)
    reference, _ = _reference(make_fx(fn)(*inputs))
    save_reference(reference, tmp_path / "reference.json")
    outputs = fn(*inputs)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    request = {
        "reference_path": str(tmp_path / "reference.json"),
        "samples_path": None,
        "backend": backend_name,
        "backend_version": backend.version,
        "proposal_schema_sha256": digest(backend.proposal_schema),
        "inputs": [_describe(spec) for spec in _boundaries(*inputs)],
        "outputs": [_describe(spec) for spec in _boundaries(*outputs)],
        "objective": "candidate",
        "diagnostics": False,
        "num_tests": 3,
        "benchmark_iters": 5,
        "benchmark_repeats": 3,
        "check_nonfinite": True,
        "rtol": None,
        "atol": None,
        "validation_seed": 20261107,
        "validation_phase": phase,
        "screening": "launch",
        "check_baseline": False,
        "compile_options": _compile_options(None),
        "candidate": candidate,
    }
    directory = tmp_path / phase
    directory.mkdir()
    (directory / "evaluation.json").write_text(json.dumps(request))
    process = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "torch_tensorrt.dynamo.region_compilation._generation._worker",
            "--request",
            str(directory / "evaluation.json"),
            "--result",
            str(directory / "result.json"),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result = directory / "result.json"
    assert result.is_file(), process.stderr[-4000:]
    return json.loads(result.read_text())


@requires_gpu_plugins
@pytest.mark.parametrize(
    "phase,execution", [("search", "launch"), ("holdout", "engine")]
)
def test_worker_accepts_the_cutedsl_guide_example(tmp_path, phase, execution):
    require_cutedsl()
    a, b = (torch.randn(1000, device="cuda") for _ in range(2))
    result = _evaluate(
        tmp_path, "cutedsl", torch.ops.aten.add.Tensor, (a, b), cute_add(), phase
    )
    assert result["status"] == "passed", result
    assert result["execution"] == execution


@requires_gpu_plugins
def test_worker_attributes_a_cutedsl_shared_memory_overrun_to_correctness(tmp_path):
    require_cutedsl()
    source = """import cutlass
import cutlass.cute as cute

@cute.kernel
def rev(arg0, out0, BLOCK: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BLOCK), byte_alignment=16)
    buf[tidx] = arg0[bidx * BLOCK + tidx]
    cute.arch.barrier()
    out0[bidx * BLOCK + tidx] = buf[BLOCK - 1 - tidx]
"""
    candidate = {
        "source": source,
        "kernel_name": "rev",
        "grid": [4],
        "block": [256],
        "shared_mem_bytes": 64,
        "constants": [{"name": "BLOCK", "value": 256}],
    }
    result = _evaluate(
        tmp_path,
        "cutedsl",
        lambda v: v.view(4, 256).flip(-1).reshape(1024),
        (torch.randn(1024, device="cuda"),),
        candidate,
        "search",
    )
    assert result["status"] == "error" and result["stage"] == "correctness", result


CUDA_TYPES = {
    torch.float16: ("cuda_fp16.h", "__half", "__half2float", "__float2half_rn"),
    torch.bfloat16: (
        "cuda_bf16.h",
        "__nv_bfloat16",
        "__bfloat162float",
        "__float2bfloat16_rn",
    ),
}


@requires_gpu_plugins
@pytest.mark.parametrize("dtype", list(CUDA_TYPES), ids=["fp16", "bf16"])
@pytest.mark.parametrize("phase", ["search", "holdout"])
def test_worker_accepts_half_precision_cuda_cpp(tmp_path, dtype, phase):
    require_cuda_cpp_toolchain()
    header, ctype, load, store = CUDA_TYPES[dtype]
    source = (
        f"#include <{header}>\n"
        f'extern "C" __global__ void add(const {ctype}* arg0, const {ctype}* arg1, {ctype}* out0) {{\n'
        "    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        f"    if (i < 1024) out0[i] = {store}({load}(arg0[i]) + {load}(arg1[i]));\n"
        "}\n"
    )
    candidate = {
        "source": source,
        "kernel_name": "add",
        "grid": [4],
        "block": [256],
        "shared_mem_bytes": 0,
    }
    a, b = (torch.randn(1024, device="cuda", dtype=dtype) for _ in range(2))
    result = _evaluate(
        tmp_path,
        "cuda_cpp",
        torch.ops.aten.add.Tensor,
        (a, b),
        copy.deepcopy(candidate),
        phase,
    )
    assert result["status"] == "passed", result


@requires_gpu_plugins
def test_cutile_direct_launch_compiles_without_tma(tmp_path, monkeypatch):
    """The screening launch must run the plugin's code, which never uses TMA."""
    ct = pytest.importorskip("cuda.tile")
    from torch_tensorrt.kernels._cutile import _tma_hints

    operations = _tma_hints()
    if operations is None:
        pytest.skip("this cuda-tile release has no view load/store hook")
    forwarded = []

    class View:
        allow_tma = True

    def generate_bytecode(self, ctx):
        forwarded.append(self.allow_tma)

    def fake_launch(stream, grid, kernel, args):
        # cuTile generates each view load's and store's code while it compiles.
        for operation in operations:
            operation.generate_bytecode(View(), None)

    for operation in operations:
        monkeypatch.setattr(operation, "generate_bytecode", generate_bytecode)
    monkeypatch.setattr(ct, "launch", fake_launch)
    candidate = {
        "source": CUTILE_ADD,
        "kernel_name": "add",
        "grid": [2],
        "constants": [{"name": "N", "value": 256}, {"name": "BLOCK", "value": 128}],
    }
    x, y = torch.randn(256, device="cuda"), torch.empty(256, device="cuda")
    get_backend("cutile").launch(candidate, [x], [y], tmp_path)
    assert forwarded == [False, False]
    assert all(op.generate_bytecode is generate_bytecode for op in operations)


# ct.gather and ct.scatter load and store through pointers; tileiras rejects
# the allow_tma hint on those, so the TMA-free launch must leave them alone.
CUTILE_GATHER = """import cuda.tile as ct

@ct.kernel
def add_one(arg0, out0, TILE: ct.Constant[int]):
    index = ct.bid(0) * TILE + ct.arange(TILE, dtype=ct.int32)
    ct.scatter(out0, index, ct.gather(arg0, index) + 1.0)
"""


def cutile_gather():
    return {
        "source": CUTILE_GATHER,
        "kernel_name": "add_one",
        "grid": [16],
        "constants": [{"name": "TILE", "value": 256}],
    }


def require_cutile():
    pytest.importorskip("cuda.tile")
    from torch_tensorrt.kernels._cutile import validate_cutile_toolchain

    try:
        validate_cutile_toolchain("generation frontend tests")
    except RuntimeError as error:
        pytest.skip(str(error))


@requires_gpu_plugins
def test_cutile_direct_launch_compiles_gather_and_scatter(tmp_path):
    require_cutile()
    backend = get_backend("cutile")
    candidate = backend.validate_candidate(cutile_gather(), 1, 1)
    x = torch.randn(4096, device="cuda")
    y = torch.empty_like(x)
    backend.launch(candidate, [x], [y], tmp_path)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)


@requires_gpu_plugins
def test_worker_accepts_a_cutile_gather_and_scatter_kernel(tmp_path):
    """Every screening, guard-band and timed launch compiles without TMA."""
    require_cutile()
    x = torch.randn(4096, device="cuda")
    result = _evaluate(
        tmp_path, "cutile", lambda v: v + 1, (x,), cutile_gather(), "search"
    )
    assert result["status"] == "passed", result
    assert result["execution"] == "launch"
