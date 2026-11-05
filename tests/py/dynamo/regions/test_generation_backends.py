# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend contracts and reconstruction across the evaluator process boundary."""

import os
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torch_tensorrt import kernels
from torch_tensorrt.dynamo.region_compilation import _plugins
from torch_tensorrt.dynamo.region_compilation._generation import _cuda
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend
from torch_tensorrt.dynamo.region_compilation._generation._cuda import _validate_ptx
from torch_tensorrt.region import RegionError

from ._generation_utils import (
    make_registration,
    require_cuda_cpp_toolchain,
    validate_candidate,
)

CUDA_SOURCE = """extern "C" __global__ void generated(const float* arg0, float* out0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 257) out0[index] = arg0[index] + 1.0f;
}
"""


def boundary(dtype=torch.float32):
    return _plugins._TensorBoundary((257,), dtype, torch.device("cuda:0"))


def candidate(**updates):
    return (
        dict(
            source=CUDA_SOURCE,
            kernel_name="generated",
            grid=[3],
            block=[128],
            shared_mem_bytes=0,
        )
        | updates
    )


def test_backend_schemas_only_expose_applicable_launch_fields():
    triton = get_backend("triton").proposal_schema
    cutile = get_backend("cutile").proposal_schema
    cuda = get_backend("cuda_cpp").proposal_schema
    assert "num_warps" in triton["properties"]
    assert "num_warps" not in cutile["properties"]
    assert "constants" in cutile["properties"]
    assert set(cuda["properties"]) == set(candidate())
    cuda["properties"].clear()
    assert get_backend("cuda_cpp").proposal_schema["properties"]
    assert get_backend("triton") is get_backend("triton")


def test_backend_schemas_encode_validator_limits_for_structured_output():
    triton = get_backend("triton").proposal_schema["properties"]
    cuda = get_backend("cuda_cpp").proposal_schema["properties"]
    assert triton["num_warps"]["enum"] == [None, 1, 2, 4, 8, 16, 32]
    assert triton["num_stages"]["minimum"] == 1
    assert triton["grid"]["items"]["maximum"] == 2**31 - 1
    assert cuda["block"]["items"]["maximum"] == 1024
    assert cuda["shared_mem_bytes"]["maximum"] == 49152
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

    def keywords(schema):
        assert set(schema) <= allowed, set(schema) - allowed
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert schema["required"] == list(schema["properties"])
        for child in schema.get("properties", {}).values():
            keywords(child)
        if "items" in schema:
            keywords(schema["items"])

    for name in ("triton", "cutile", "cuda_cpp"):
        keywords(get_backend(name).proposal_schema)


def test_backend_extension_is_resolved_in_a_fresh_interpreter(tmp_path, monkeypatch):
    (tmp_path / "test_kernel_backend.py").write_text(
        "from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend\n"
        "backend = get_backend('triton')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "test_kernel_backend:backend"
    assert get_backend(name).name == "triton"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend; "
            f"assert get_backend({name!r}).name == 'triton'",
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name", ["cuda_python", "numba", "invalid:path.extra", "a:b:c", None]
)
def test_unknown_backend_fails_with_actionable_contract(name):
    with pytest.raises(ValueError, match="backend|Backend"):
        get_backend(name)


def test_extension_must_implement_backend_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "invalid_kernel_backend", SimpleNamespace(backend=object())
    )
    with pytest.raises(ValueError, match="GenerationBackend"):
        get_backend("invalid_kernel_backend:backend")


@pytest.mark.parametrize(
    "updates",
    [
        {"grid": [False]},
        {"grid": [1, 65536]},
        {"block": [1024, 2]},
        {"block": [1, 1, 65]},
        {"block": [0]},
        {"shared_mem_bytes": True},
        {"shared_mem_bytes": 49153},
        {"num_warps": 4},
        {"source": CUDA_SOURCE.replace("const float* arg0", "int arg0")},
        {"source": CUDA_SOURCE.replace("float* out0", "const float* out0")},
        {"source": CUDA_SOURCE.replace("arg0, float* out0", "out0, float* arg0")},
        {"source": "#include <stdio.h>\n" + CUDA_SOURCE},
        {"source": "#define float int\n" + CUDA_SOURCE},
        {"source": CUDA_SOURCE + CUDA_SOURCE.replace("generated", "second")},
    ],
)
def test_cuda_rejects_unsupported_abi_and_launch_metadata(updates):
    with pytest.raises(ValueError):
        validate_candidate(candidate(**updates), "cuda_cpp", 1, 1)


def _kernel_line(line):
    return CUDA_SOURCE.replace("    int index", f"    {line}\n    int index")


@pytest.mark.parametrize(
    "source,match",
    [
        # NVRTC has no C/C++ standard headers.
        ("#include <math.h>\n" + CUDA_SOURCE, "math.h"),
        ("#include <cmath>\n" + CUDA_SOURCE, "cmath"),
        ("#include <stdint.h>\n" + CUDA_SOURCE, "stdint.h"),
        # Line splicing happens before the directive is checked.
        ("#inc\\\nlude <math.h>\n" + CUDA_SOURCE, "math.h"),
        ("#pragma once\n" + CUDA_SOURCE, "#pragma unroll"),
        ("%:define BUMP 1\n" + CUDA_SOURCE, "digraphs"),
        # Literals cannot hide a comment opener from the asm and directive checks.
        (
            _kernel_line(
                'const char* a = "/*"; asm volatile("membar.gl;"); const char* b = "*/";'
            ),
            "literals",
        ),
        (_kernel_line('_Pragma("unroll")'), "literals"),
        (_kernel_line("char c = '{';"), "literals"),
        (_kernel_line('asm volatile(R"(membar.gl;)");'), "literals"),
        (
            CUDA_SOURCE + "__device__ float helper(float x) { return x; }\n",
            "after the kernel",
        ),
        (
            "__device__ float helper(float x) { return x; }\n" + CUDA_SOURCE,
            "one extern",
        ),
        (CUDA_SOURCE.rstrip().removesuffix("}"), "unbalanced"),
    ],
)
def test_cuda_source_policy_matches_the_compiler_view(source, match):
    with pytest.raises(ValueError, match=match):
        validate_candidate(candidate(source=source), "cuda_cpp", 1, 1)


@pytest.mark.parametrize(
    "line",
    [
        "#pragma unroll",
        "  #  pragma unroll 4",
        "#pragma nounroll",
        '// a comment may mention "#define", asm(...), and quotes \'',
        "/* a block comment\n   may span lines and mention } or #include <cmath> */",
    ],
)
def test_cuda_accepts_unroll_pragmas_and_comments(line):
    source = '#include "cuda_fp16.h"\n' + _kernel_line(line)
    assert validate_candidate(candidate(source=source), "cuda_cpp", 1, 1)["source"] == (
        source
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("header", sorted(_cuda._HEADERS))
def test_cuda_allowed_headers_and_pragmas_compile_under_nvrtc(header):
    pytest.importorskip("cuda.core")
    from torch_tensorrt.kernels._nvrtc import compile_to_ptx

    loops = "".join(
        f"#pragma {pragma}\n    for (int i = 0; i < 4; ++i) value += 0.25f;\n"
        for pragma in ("unroll", "unroll 2", "nounroll")
    )
    source = f"#include <{header}>\n" + _kernel_line(
        f"float value = 0.0f;\n{loops}"
    ).replace("arg0[index] + 1.0f", "arg0[index] + value")
    validate_candidate(candidate(source=source), "cuda_cpp", 1, 1)
    ptx, _, _ = compile_to_ptx(source, "generated", None)
    _validate_ptx(ptx, "generated", 2, (128, 1, 1))


def test_cuda_normalized_launch_lists_are_independent():
    original = candidate()
    normalized = validate_candidate(original, "cuda_cpp", 1, 1)
    normalized["block"].append(1)
    assert original["block"] == [128]


def test_cuda_compiled_entry_contract_is_checked():
    ptx = b".visible .entry generated(.param .u64 .ptr .align 1 a, .param .u64 b) {}"
    _validate_ptx(ptx, "generated", 2, (128, 1, 1))
    for invalid in (
        ptx.replace(b".u64 b", b".u32 b"),
        ptx.replace(b"generated", b"other"),
        ptx + b"\n.visible .entry second() {}",
        ptx.replace(b") {", b") .reqntid 64 {"),
    ):
        with pytest.raises(ValueError):
            _validate_ptx(invalid, "generated", 2, (128, 1, 1))


def test_cuda_pointer_dtype_must_match_captured_boundary(tmp_path):
    spec = boundary(torch.float16)
    with pytest.raises(ValueError, match="boundary dtype"):
        make_registration(candidate(), "cuda_cpp", [spec], [spec], tmp_path)


@pytest.mark.parametrize("cuda_core_major", [0, 1])
def test_cuda_region_reuses_validated_ptx_and_guards_engine_bindings(
    tmp_path, monkeypatch, cuda_core_major
):
    import tensorrt as trt

    from torch_tensorrt.kernels import _nvrtc, _ops, _register

    # TensorRT-RTX builds have no QDP Python module.
    trtp = pytest.importorskip("tensorrt.plugin")

    ptx = b".visible .entry generated(.param .u64 arg0, .param .u64 out0) { ret; }"
    compilations = []

    # cuda-core 0.x exposes kernel attributes as methods, 1.x as properties.
    attributes = dict(max_threads_per_block=1024, shared_size_bytes=0)
    if cuda_core_major == 0:
        attributes = {
            name: (lambda value=value: value) for name, value in attributes.items()
        }

    def compile_source(source, name, include_paths, *, load_kernel=False):
        compilations.append((source, name, load_kernel))
        return ptx, None, None

    loaded = []

    def load_kernel(code, name):
        # The eager kernel comes from the same PTX that the plugin embeds.
        loaded.append((code, name))
        return SimpleNamespace(attributes=SimpleNamespace(**attributes))

    monkeypatch.setattr(_nvrtc, "compile_to_ptx", compile_source)
    monkeypatch.setattr(_cuda, "_driver_rejection", lambda code, name: None)
    monkeypatch.setattr(_cuda, "_ptx_kernel", load_kernel)
    monkeypatch.setattr(_cuda, "_KERNELS", {})
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            max_threads_per_block=1024, shared_memory_per_block=49152
        ),
    )
    spec = boundary()
    registration = make_registration(candidate(), "cuda_cpp", [spec], [spec], tmp_path)
    assert isinstance(registration, partial) and registration.func is kernels.ptx_op
    assert registration.keywords["ptx"] is ptx

    child = torch.fx.symbolic_trace(lambda x: x + 1)
    monkeypatch.setattr(_plugins, "_boundaries", lambda *args: ([spec], [spec]))
    monkeypatch.setattr(_ops, "_require_qdp_plugin", lambda: None)
    captured = {}

    def capture_registration(**options):
        captured.update(options)
        namespace, name = options["op_name"].split("::")
        monkeypatch.setattr(
            getattr(torch.ops, namespace),
            name,
            SimpleNamespace(default=object()),
            raising=False,
        )

    monkeypatch.setattr(
        _register, "register_precompiled_qdp_plugin", capture_registration
    )
    _plugins.register_region_plugin(child, registration, region_id="generated_cuda")
    assert compilations == [(CUDA_SOURCE, "generated", False)]
    assert loaded == [(ptx, "generated")]
    assert captured["ptx"] is ptx
    assert captured["schema"] == "(Tensor arg0) -> Tensor"
    assert captured["supports_dynamic_shapes"] is False
    assert callable(captured["capability_validator"])
    actual = SimpleNamespace(dtype=trt.float32)
    wrong = SimpleNamespace(dtype=trt.float16)
    for inputs, outputs in (([wrong], [actual]), ([actual], [wrong])):
        with pytest.raises(ValueError, match="descriptor has dtype"):
            captured["aot_fn"](inputs, outputs, 0)
    for inputs, outputs in (([], [actual]), ([actual], [])):
        with pytest.raises(ValueError, match="expected 1"):
            captured["aot_fn"](inputs, outputs, 0)
    monkeypatch.setattr(trtp, "KernelLaunchParams", SimpleNamespace)
    launch, extras = captured["aot_fn"]([actual], [actual], 0)
    assert (launch.grid_x, launch.grid_y, launch.grid_z) == (3, 1, 1)
    assert (launch.block_x, launch.block_y, launch.block_z) == (128, 1, 1)
    assert launch.shared_mem == 0 and len(extras) == 0


def test_cuda_toolchain_rejected_before_any_author_call(monkeypatch):
    # A driver older than NVRTC cannot load its PTX; TensorRT would only fail when
    # the engine runs, after every candidate passed its eager checks.
    pytest.importorskip("cuda.core")
    rejection = "CUDA_ERROR_UNSUPPORTED_PTX_VERSION"
    monkeypatch.setattr(_cuda, "_toolchain_rejection", lambda device: rejection)
    with pytest.raises(RegionError, match=rejection):
        get_backend("cuda_cpp").target_metadata(torch.device("cuda", 0))


def test_cuda_compile_rejects_ptx_the_driver_cannot_load(monkeypatch):
    from torch_tensorrt.kernels import _nvrtc

    monkeypatch.setattr(
        _nvrtc, "compile_to_ptx", lambda source, name, include_paths: (b"ptx", 0, None)
    )
    monkeypatch.setattr(
        _cuda, "_driver_rejection", lambda code, name: "CUDA_ERROR_INVALID_PTX"
    )
    monkeypatch.setattr(_cuda, "_KERNELS", {})
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_PTX"):
        _cuda._compiled(CUDA_SOURCE, "generated", torch.device("cuda", 0))


def _scripted_probe(monkeypatch, *results):
    """Return each of ``results`` from successive driver loads of the probe."""
    from torch_tensorrt.kernels import _nvrtc

    loads = []

    def load(code, name):
        loads.append(name)
        return results[len(loads) - 1]

    monkeypatch.setattr(
        _nvrtc, "compile_to_ptx", lambda source, name, include_paths: (b"ptx", 0, None)
    )
    monkeypatch.setattr(_cuda, "_driver_rejection", load)
    monkeypatch.setattr(_cuda, "_TOOLCHAIN", {})
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return loads


def test_cuda_toolchain_probe_retries_after_a_transient_driver_error(monkeypatch):
    # A full GPU must not condemn the toolchain for the rest of the process.
    pytest.importorskip("cuda.core")
    loads = _scripted_probe(monkeypatch, "CUDA_ERROR_OUT_OF_MEMORY", None)
    device = torch.device("cuda", 0)
    with pytest.raises(
        RuntimeError, match=r"could not load a probe kernel \(CUDA_ERROR_OUT_OF_MEMORY"
    ):
        _cuda.verify_toolchain(device)
    _cuda.verify_toolchain(device)
    _cuda.verify_toolchain(device)
    assert len(loads) == 2


@pytest.mark.parametrize(
    "encoded, version, rejection, advice",
    [
        (13000, "13.0", "CUDA_ERROR_UNSUPPORTED_PTX_VERSION", True),
        (13030, "13.3", "CUDA_ERROR_JIT_COMPILATION_DISABLED", False),
    ],
)
def test_cuda_toolchain_rejection_is_remembered_and_names_both_versions(
    monkeypatch, encoded, version, rejection, advice
):
    pytest.importorskip("cuda.core")
    from cuda.bindings import driver, nvrtc

    loads = _scripted_probe(monkeypatch, rejection)
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: (0, encoded))
    monkeypatch.setattr(nvrtc, "nvrtcVersion", lambda: (0, 13, 2))
    for _ in range(2):
        with pytest.raises(RegionError) as error:
            _cuda.verify_toolchain(torch.device("cuda", 0))
        message = str(error.value)
        assert f"NVRTC 13.2 PTX; the CUDA {version} driver reported {rejection}." in (
            message
        )
        # Only a driver older than NVRTC calls for a different version.
        assert ("Install an NVRTC no newer than the driver" in message) is advice
    assert len(loads) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_registration_compiles_real_source_and_derives_trusted_launch(
    tmp_path, monkeypatch
):
    require_cuda_cpp_toolchain()
    trtp = pytest.importorskip("tensorrt.plugin")

    spec = boundary()
    registration = make_registration(candidate(), "cuda_cpp", [spec], [spec], tmp_path)
    assert isinstance(registration, partial) and registration.func is kernels.ptx_op
    _validate_ptx(registration.keywords["ptx"], "generated", 2, (128, 1, 1))
    monkeypatch.setattr(trtp, "KernelLaunchParams", SimpleNamespace)
    params, extras = registration.keywords["aot_fn"]([], [], None)
    assert params.grid_x == 3
    assert params.block_x == 128
    assert len(extras) == 0
    assert len(list(tmp_path.glob("*.cu"))) == 1


@pytest.mark.parametrize(
    "registration",
    [partial(kernels.cutedsl_op, kernel=None), kernels.triton_op, object()],
)
def test_region_rejects_unsupported_registration_forms(monkeypatch, registration):
    child = torch.fx.symbolic_trace(lambda x: x + 1)
    spec = boundary()
    monkeypatch.setattr(_plugins, "_boundaries", lambda *args: ([spec], [spec]))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    with pytest.raises(RegionError, match="KernelSpec or a functools.partial"):
        _plugins.register_region_plugin(child, registration, region_id="unsupported")


def test_toolchain_suite_selects_tests_from_every_file_it_lists(monkeypatch):
    # CI reruns these files with cuTile and cuda-core installed, selecting only
    # their toolchain cases; a listed file without one would select nothing.
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root))
    from tests.ci.suites import by_name

    suite = by_name("regions-generation").for_variant("standard")
    collection = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-k",
            suite["keyword"],
            *suite["paths"],
        ],
        cwd=root / suite["cwd"],
        capture_output=True,
        text=True,
    )
    assert collection.returncode == 0, collection.stdout + collection.stderr
    for path in suite["paths"]:
        assert f"{path}::" in collection.stdout, path
