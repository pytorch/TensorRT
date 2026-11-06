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

from ._generation_utils import make_registration, validate_candidate

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
    proposal = dict(
        source=TRITON_SOURCE if backend == "triton" else CUTILE_SOURCE,
        kernel_name="generated",
        constants=[{"name": "N", "value": 257}, {"name": "BLOCK", "value": 128}],
        grid=[3],
    )
    if backend == "triton":
        proposal.update(num_warps=4, num_stages=None)
    return proposal


def boundary():
    return _plugins._TensorBoundary((257,), torch.float32, torch.device("cuda:0"))


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


def test_cutile_requires_matching_constant_types():
    proposal = candidate("cutile")
    proposal["constants"][0]["value"] = 257.0
    with pytest.raises(ValueError, match="annotation"):
        validate_candidate(proposal, "cutile", 1, 1)


@pytest.mark.parametrize("options", [{"num_warps": 4}, {"num_warps": None}])
def test_cutile_candidates_contain_exactly_the_cutile_schema_fields(options):
    from torch_tensorrt.dynamo.region_compilation._generation._backends import (
        get_backend,
    )

    fields = set(get_backend("cutile").proposal_schema["properties"])
    assert set(validate_candidate(candidate("cutile"), "cutile", 1, 1)) == fields
    with pytest.raises(ValueError, match="exactly"):
        validate_candidate(dict(candidate("cutile"), **options), "cutile", 1, 1)


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
    spec = boundary()
    registration = make_registration(
        candidate(backend), backend, [spec], [spec], tmp_path
    )
    assert registration.func is getattr(kernels, f"{backend}_op")
    assert registration.keywords["grid"]([], []) == (3,)
    # Region registration derives the pointer signature from the boundaries.
    assert "signature" not in registration.keywords
    if backend == "triton":
        signature = _plugins._triton_signature(
            registration, [spec], [spec], "generated"
        )
        assert signature == {"arg0": "*fp32", "out0": "*fp32"}
    else:
        assert registration.keywords["constants"] == {"N": 257, "BLOCK": 128}
        assert registration.keywords["ndim"] == 1
    assert len(list(tmp_path.glob("*.py"))) == 1
    assert registration.keywords["kernel"] is not None
    other = make_registration(
        candidate(backend), backend, [spec], [spec], tmp_path / "other"
    )
    kernel = other.keywords["kernel"]
    function = kernel.fn if backend == "triton" else kernel._pyfunc
    assert Path(inspect.getsourcefile(function)).parent == tmp_path / "other"


@pytest.mark.parametrize(
    "signature,names",
    [
        (None, ("arg0",)),
        ({"x": torch.float32, "out": torch.float32}, ("x",)),
    ],
)
def test_cutile_region_callback_exposes_the_kernel_array_names(
    monkeypatch, signature, names
):
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
        assert info.tensor_arg_names == names
        library.define(op_name.split("::")[1] + options["schema"])

    monkeypatch.setattr(kernels, "cutile_op", register)
    registration = (
        partial(register, signature=signature) if signature else partial(register)
    )
    op = _plugins.register_region_plugin(child, registration, region_id="generated")
    assert op is not None
    assert tuple(inspect.signature(captured["meta_fn"]).parameters) == names
    assert captured["signature"] == (
        signature or {"arg0": torch.float32, "out0": torch.float32}
    )
    assert captured["supports_dynamic_shapes"] is False
    assert callable(captured["capability_validator"])


def _with_body(backend, line):
    source = TRITON_SOURCE if backend == "triton" else CUTILE_SOURCE
    anchor = "    tl.store" if backend == "triton" else "    ct.store"
    return dict(
        candidate(backend), source=source.replace(anchor, f"    {line}\n{anchor}")
    )


@pytest.mark.parametrize(
    "backend,line",
    [
        ("triton", "values = values.cast(tl.float32)"),
        ("triton", "values = values.exp().ravel()"),
        ("triton", "values = tl.math.exp(values)"),
        ("triton", "total = tl.load(arg0 + offsets, offsets < N).cumsum(0)"),
        ("triton", "blocks = triton.cdiv(N, BLOCK)"),
        ("cutile", "values = values.transpose()"),
        ("cutile", "first = values.item()"),
        ("cutile", "view = arg0.slice(0, 0, BLOCK)"),
        ("cutile", "tiles = arg0.tiled_view((BLOCK,))"),
    ],
)
def test_dsl_accepts_installed_tensor_methods_and_math_functions(backend, line):
    pytest.importorskip("triton" if backend == "triton" else "cuda.tile")
    proposal = _with_body(backend, line)
    assert validate_candidate(proposal, backend, 1, 1)["source"] == proposal["source"]


@pytest.mark.parametrize(
    "backend,line",
    [
        ("triton", "tl.core.os.system('true')"),
        ("triton", "triton.runtime.driver.active.get_current_device()"),
        ("triton", "values = tl.extra.cuda.libdevice.exp(values)"),
        ("triton", "values = triton.language.exp(values)"),
        ("triton", "values = values.system()"),
        ("triton", "values = getattr(tl, 'exp')(values)"),
        ("triton", "values = tl.load(arg0 + offsets)()"),
        ("triton", "values = ct.load(arg0, index=(0,), shape=(BLOCK,))"),
        ("cutile", "ct.compilation.export_kernel(generated)"),
        ("cutile", "values = tl.exp(values)"),
        ("cutile", "values = open('/etc/passwd')"),
    ],
)
def test_dsl_rejects_calls_outside_the_kernel_language(backend, line):
    with pytest.raises(ValueError, match="supported .* DSL"):
        validate_candidate(_with_body(backend, line), backend, 1, 1)
