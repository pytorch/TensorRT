# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the generated-source boundary without running a model or GPU job."""

import inspect
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import pytest
import torch

from torch_tensorrt import kernels
from torch_tensorrt.dynamo.region_compilation import _plugins
from torch_tensorrt.dynamo.region_compilation._generation._candidate import (
    make_registration,
    validate_candidate,
)

TRITON_SOURCE = """import triton
import triton.language as tl

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    tl.store(out0 + offsets, values + 1, offsets < N)
"""
CUTILE_SOURCE = """import cuda.tile as ct

@ct.kernel
def generated(arg0, out0, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    values = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,))
    ct.store(out0, index=(ct.bid(0),), tile=values + 1)
"""


def candidate(backend="triton"):
    return dict(
        source=TRITON_SOURCE if backend == "triton" else CUTILE_SOURCE,
        kernel_name="generated",
        constants=[{"name": "N", "value": 257}, {"name": "BLOCK", "value": 128}],
        grid=[3],
        num_warps=4 if backend == "triton" else None,
        num_stages=None,
    )


@pytest.mark.parametrize("backend", ["triton", "cutile"])
def test_normalized_candidate_has_independent_containers(backend):
    original = candidate(backend)
    normalized = validate_candidate(original, backend, 1, 1)
    assert normalized == original
    normalized["grid"].append(2)
    normalized["constants"][0]["value"] = 1
    assert original["grid"] == [3]
    assert original["constants"][0]["value"] == 257


@pytest.mark.parametrize(
    "source",
    [
        TRITON_SOURCE + "print('host execution')\n",
        TRITON_SOURCE.replace("import triton\n", "import os\nimport triton\n"),
        TRITON_SOURCE.replace("@triton.jit", "@triton.jit()"),
        TRITON_SOURCE.replace("arg0, out0", "out0, arg0"),
        TRITON_SOURCE.replace("BLOCK: tl.constexpr", "BLOCK: tl.constexpr = 128"),
        TRITON_SOURCE.replace("BLOCK: tl.constexpr", "BLOCK: int"),
        TRITON_SOURCE + "\ndef another():\n    pass\n",
        TRITON_SOURCE + "    import torch\n",
        TRITON_SOURCE + "    eval('1')\n",
        TRITON_SOURCE + "    values.__class__\n",
        TRITON_SOURCE + "    callback = lambda: 1\n",
        TRITON_SOURCE
        + "    tl.inline_asm_elementwise('asm', constraints='', args=[], dtype=tl.float32, is_pure=True, pack=1)\n",
    ],
)
def test_reject_host_code_and_wrong_kernel_contract(source):
    with pytest.raises(ValueError):
        validate_candidate(dict(candidate(), source=source), "triton", 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid", [True]),
        ("grid", [1, 65536]),
        ("grid", [0]),
        ("grid", "lambda: 1"),
        ("num_warps", 3),
        ("num_stages", True),
        ("constants", [{"name": "N", "value": float("nan")}]),
        ("constants", [{"name": "N", "value": "257"}]),
        ("constants", [{"name": "arg0", "value": 257}]),
        ("constants", [{"name": "N", "value": 1}, {"name": "N", "value": 2}]),
        ("kernel_name", "__import__"),
    ],
)
def test_reject_invalid_launch_metadata(field, value):
    with pytest.raises(ValueError):
        validate_candidate(dict(candidate(), **{field: value}), "triton", 1, 1)


def test_cutile_requires_matching_constant_types_and_no_triton_options():
    proposal = candidate("cutile")
    proposal["constants"][0]["value"] = 257.0
    with pytest.raises(ValueError, match="annotation"):
        validate_candidate(proposal, "cutile", 1, 1)
    with pytest.raises(ValueError, match="compile options"):
        validate_candidate(dict(candidate("cutile"), num_warps=4), "cutile", 1, 1)


def test_cutile_schema_does_not_require_triton_compile_options():
    proposal = candidate("cutile")
    del proposal["num_warps"]
    del proposal["num_stages"]
    normalized = validate_candidate(proposal, "cutile", 1, 1)
    assert normalized == candidate("cutile")


def test_triton_libdevice_supports_explicit_rounding():
    source = TRITON_SOURCE.replace(
        "import triton.language as tl",
        "import triton.language as tl\nimport triton.language.extra.cuda.libdevice as libdevice",
    ).replace("values + 1", "libdevice.add_rn(values, 1.0)")
    assert (
        validate_candidate(dict(candidate(), source=source), "triton", 1, 1)["source"]
        == source
    )


@pytest.mark.parametrize("backend", ["triton", "cutile"])
def test_registration_reuses_frontend_and_derives_grid_and_dtypes(backend, tmp_path):
    pytest.importorskip("triton" if backend == "triton" else "cuda.tile")
    spec = {"dtype": "float32", "shape": [257]}
    registration = make_registration(
        candidate(backend), backend, [spec], [spec], tmp_path
    )
    assert registration.func is getattr(kernels, f"{backend}_op")
    assert registration.keywords["grid"]([], []) == (3,)
    signature = registration.keywords["signature"]
    assert list(signature) == ["arg0", "out0"]
    assert list(signature.values()) == (
        ["*fp32", "*fp32"] if backend == "triton" else [torch.float32, torch.float32]
    )
    assert len(list(tmp_path.glob("*.py"))) == 1
    assert registration.keywords["kernel"] is not None
    other = make_registration(
        candidate(backend), backend, [spec], [spec], tmp_path / "other"
    )
    kernel = other.keywords["kernel"]
    function = kernel.fn if backend == "triton" else kernel._pyfunc
    assert Path(inspect.getsourcefile(function)).parent == tmp_path / "other"


def test_cutile_region_callback_exposes_the_captured_tensor_names(monkeypatch):
    from torch_tensorrt.kernels._cutile import analyze_tensor_schema

    child = torch.fx.symbolic_trace(lambda x: x.relu())
    boundary = _plugins._TensorBoundary((8,), torch.float32, torch.device("cuda:0"))
    monkeypatch.setattr(_plugins, "_boundaries", lambda *args: ([boundary], [boundary]))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    library = torch.library.Library("torch_tensorrt_region_fused", "FRAGMENT")
    captured = {}

    def register(op_name, **options):
        captured.update(options)
        info = analyze_tensor_schema(op_name, options["meta_fn"], options["schema"])
        assert info.tensor_arg_names == ("arg0",)
        library.define(op_name.split("::")[1] + options["schema"])

    monkeypatch.setattr(kernels, "cutile_op", register)
    op = _plugins.register_region_plugin(
        child, partial(register), region_id="generated"
    )
    assert op is not None
    assert list(inspect.signature(captured["meta_fn"]).parameters) == ["arg0"]
    assert captured["supports_dynamic_shapes"] is False
    assert callable(captured["capability_validator"])
