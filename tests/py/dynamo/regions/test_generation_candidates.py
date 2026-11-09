# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the generated-source boundary without running a model or GPU job."""

import ast
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


def scaled_cutile(factor):
    proposal = candidate("cutile")
    proposal["source"] = CUTILE_SOURCE.replace(
        "BLOCK: ct.Constant[int])",
        "BLOCK: ct.Constant[int], FACTOR: ct.Constant[float])",
    ).replace("values + 1", "values * FACTOR")
    proposal["constants"].append({"name": "FACTOR", "value": factor})
    return proposal


@pytest.mark.parametrize("factor", [2.0, 2], ids=["float", "whole"])
def test_cutile_float_constants_accept_whole_numbers(factor):
    """JSON need not keep 2.0 apart from 2: the claude CLI rewrites its reply
    in JavaScript, where they are one number, and returns 2."""
    proposal = scaled_cutile(factor)
    normalized = validate_candidate(proposal, "cutile", 1, 1)
    value = normalized["constants"][-1]["value"]
    assert type(value) is float and value == 2.0
    assert proposal["constants"][-1]["value"] is factor


@pytest.mark.parametrize(
    "factor,match",
    [(True, r"ct\.Constant\[bool\]"), (10**400, "too large for a float")],
    ids=["bool", "huge"],
)
def test_cutile_float_constants_reject_bools_and_unrepresentable_ints(factor, match):
    with pytest.raises(ValueError, match=match):
        validate_candidate(scaled_cutile(factor), "cutile", 1, 1)


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


# Kernel compilers run kernel code as Python while compiling, so a kernel that
# reaches a module or a builtin such as open can act on the host and still
# compute the right output.
@pytest.mark.parametrize(
    "line,match",
    [
        # Uses other than calls of names past the DSL.
        (
            "tl.static_print('ENV', tl.extra.pkgutil.os.environ)",
            r"'tl\.extra\.pkgutil\.os\.environ' is not allowed",
        ),
        (
            "(tl.core.inspect.functools,)[0].reduce("
            "tl.core.inspect.os.rename, ('dst',), 'src')",
            # Without Triton installed, as on Windows, the names past the DSL
            # cannot be resolved and the call check rejects the line first.
            "not modules or other objects|calls must use the supported triton DSL",
        ),
        ("dtype = triton.language.float32", "supported triton DSL"),
        # DSL modules only name their functions, types and constants.
        ("modules = (tl,)", "DSL module 'tl' only"),
        ("tl.static_print(libdevice)", "DSL module 'libdevice' only"),
        # Builtins other than the permitted ones, even when not called.
        ("tl.static_print(eval)", "'eval' is not a parameter"),
        ("handle = open", "'open' is not a parameter"),
        ("tl.static_print(generated)", "'generated' is not a parameter"),
        # Assignments that would change DSL objects.
        ("tl.load = tl.store", "cannot assign or delete attributes"),
        ("values.shape = (BLOCK,)", "cannot assign or delete attributes"),
        ("del values.dtype", "cannot assign or delete attributes"),
        ("match N:\n        case 1:\n            pass", "not Match statements"),
        (
            "try:\n        pass\n    except* ValueError:\n        pass",
            "not TryStar statements",
        ),
    ],
)
def test_kernel_code_names_only_its_dsl_helpers_and_builtins(line, match):
    with pytest.raises(ValueError, match=match):
        validate_candidate(_with_body("triton", line), "triton", 1, 1)


@pytest.mark.parametrize(
    "line,match",
    [
        # Modules in the installed tl namespace: as values, they reached
        # functools.reduce and os functions through tensor method names.
        (
            "(tl.core,)[0].inspect.functools.reduce(abs, (1,), 2)",
            "not modules or other objects",
        ),
        ("namespace = tl.math", "not modules or other objects"),
        # Past a namespace member that is not a class, such as the instance
        # tl.float32, only data attributes are DSL names.
        ("values = tl.float32.to_ir", "not modules or other objects"),
        # tl.math imports functools.wraps, which is not Triton's own.
        ("values = tl.math.wraps(values)", r"'tl\.math\.wraps' is not allowed"),
        # A name that does not resolve yet could be a module once imported.
        ("operation = tl.undefined_op", r"'tl\.undefined_op' is not allowed"),
    ],
)
def test_installed_dsl_decides_which_names_kernels_may_use(line, match):
    pytest.importorskip("triton.language")
    with pytest.raises(ValueError, match=match):
        validate_candidate(_with_body("triton", line), "triton", 1, 1)


@pytest.mark.parametrize(
    "line,name",
    [
        # Every way to bind a name: the call check trusts these names.
        ("abs = tl.load(arg0 + offsets)", "abs"),
        ("max = 1", "max"),
        ("values, len = values, 1", "len"),
        ("*values, range = values, 1", "range"),
        ("min += 1", "min"),
        ("int: tl.constexpr = 3", "int"),
        ("for float in range(2):\n        pass", "float"),
        ("values = [values for bool in range(2)]", "bool"),
        ("(tuple := 3)", "tuple"),
        ("del abs", "abs"),
        ("tl = values", "tl"),
        ("triton = 1", "triton"),
        ("libdevice = 1", "libdevice"),
    ],
)
def test_kernel_code_cannot_rebind_builtins_or_dsl_modules(line, name):
    with pytest.raises(ValueError, match=f"cannot bind '{name}'"):
        validate_candidate(_with_body("triton", line), "triton", 1, 1)


def test_kernel_names_and_parameters_cannot_rebind_builtins():
    proposal = candidate()
    proposal["source"] = (
        TRITON_SOURCE.replace("N: tl.constexpr", "max: tl.constexpr")
    ).replace("< N", "< max")
    proposal["constants"][0]["name"] = "max"
    with pytest.raises(ValueError, match="parameter 'max' would rebind"):
        validate_candidate(proposal, "triton", 1, 1)
    renamed = dict(candidate(), kernel_name="min")
    renamed["source"] = TRITON_SOURCE.replace("def generated", "def min")
    with pytest.raises(ValueError, match="kernel_name must be a nonreserved"):
        validate_candidate(renamed, "triton", 1, 1)


# The call check trusts only the permitted builtins, so kernel code may name
# variables and constants like others where their assignment certainly applies.
@pytest.mark.parametrize(
    "backend,line",
    [
        ("triton", "sum = tl.sum(values, 0)\n    values = values + sum"),
        ("triton", "input = values + 1\n    id = tl.program_id(0)\n    values = input"),
        ("triton", "for id in range(2):\n        values = values + id"),
        (
            "triton",
            "sum = values\n    for i in range(2):\n        if N > 1:\n"
            "            sum = sum + i\n    values = sum",
        ),
        (
            "triton",
            "for i in range(2):\n        sum = values + i\n        values = sum",
        ),
        ("triton", "sum = values\n    sum += 1\n    values = sum"),
        ("triton", "pair = tuple(sum for sum in range(2) if sum)"),
        # Only CuTe DSL's compiler calls type and locals inside kernels.
        ("triton", "type = values\n    locals = type\n    values = locals"),
        ("cutile", "id = ct.bid(0)\n    sum = values + id\n    values = sum"),
        ("cutile", "type = values\n    locals = type\n    values = locals"),
    ],
)
def test_kernel_code_may_name_variables_like_other_builtins(backend, line):
    proposal = _with_body(backend, line)
    assert validate_candidate(proposal, backend, 1, 1)["source"] == proposal["source"]


def test_constants_may_be_named_like_other_builtins():
    proposal = candidate()
    proposal["source"] = TRITON_SOURCE.replace("N: tl", "round: tl").replace(
        "< N", "< round"
    )
    proposal["constants"][0]["name"] = "round"
    normalized = validate_candidate(proposal, "triton", 1, 1)
    assert normalized["constants"][0] == {"name": "round", "value": 257}


# CuTe DSL moves loop and branch bodies into nested functions, and Triton drops
# what a block assigned when it ends, so these uses could reach the builtin
# itself: a CuTe DSL kernel passed open to cute.transform_leaf this way.
@pytest.mark.parametrize(
    "line,name",
    [
        (
            "for i in range(2):\n        open = values\n    tl.static_print(open)",
            "open",
        ),
        (
            "if N > 1:\n        open = values\n    else:\n        open = values\n"
            "    tl.static_print(open)",
            "open",
        ),
        (
            "while N < 1:\n        getattr = values\n    tl.static_print(getattr)",
            "getattr",
        ),
        # A comprehension binds its targets only inside, after its first iterable.
        ("first = [0 for open in (1,)]\n    tl.static_print(open)", "open"),
        ("first = [open for open in (open,)]", "open"),
        # Uses before the assignment or after del; walrus targets do not count.
        ("tl.static_print(open)\n    open = values", "open"),
        ("open = values\n    del open\n    tl.static_print(open)", "open"),
        ("(open := values)\n    tl.static_print(open)", "open"),
        ("sum += 1", "sum"),
    ],
)
def test_kernel_code_uses_builtin_names_only_where_it_assigned_them(line, name):
    with pytest.raises(ValueError, match=f"'{name}' names a Python builtin wherever"):
        validate_candidate(_with_body("triton", line), "triton", 1, 1)


@pytest.mark.parametrize(
    "line",
    [
        "values = " + " + ".join(["values"] * 2000),
        "if N == 0:\n        pass"
        + "".join(f"\n    elif N == {i}:\n        pass" for i in range(1, 2000)),
    ],
)
def test_deep_kernel_code_is_checked_without_recursion(line):
    # Python parses a long sum or chain of elif into a tree deeper than its
    # recursion limit allows a recursive walk to go.
    proposal = _with_body("triton", line)
    try:
        ast.parse(proposal["source"])
    except RecursionError:
        pytest.skip("this Python does not parse a tree this deep")
    assert validate_candidate(proposal, "triton", 1, 1)["source"] == proposal["source"]


@pytest.mark.parametrize(
    "backend,line",
    [
        ("triton", "values = values" + ".to(tl.float32)" * 400),
        ("triton", "values = tl" + ".float32" * 400),
        ("cutile", "values = values" + ".astype(ct.float32)" * 400),
    ],
    ids=["triton-method-chain", "triton-attribute-chain", "cutile-method-chain"],
)
def test_code_nested_too_deeply_to_check_costs_an_author_turn(backend, line):
    # Checking a long method or attribute chain unparses it recursively. Code
    # nested beyond Python's recursion limit must be rejected with ValueError,
    # which costs the author a turn, rather than end the search.
    with pytest.raises(ValueError, match="too deeply|invalid Python syntax"):
        validate_candidate(_with_body(backend, line), backend, 1, 1)


@pytest.mark.parametrize(
    "backend,line",
    [
        ("triton", "zeros = tl.zeros([BLOCK], dtype=tl.float32)"),
        ("triton", "dtype = tl.float32"),
        ("triton", "bits = tl.float32.primitive_bitwidth"),
        ("triton", "SCALE: tl.constexpr = 2"),
        ("triton", "count = len(values.shape) + max(abs(N), 1)"),
        ("triton", "blocks = (triton.cdiv, triton.next_power_of_2)"),
        (
            "cutile",
            "tail = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,), "
            "padding_mode=ct.PaddingMode.ZERO)",
        ),
        ("cutile", "dtype = ct.float32"),
    ],
)
def test_kernel_code_may_name_dsl_types_constants_and_enum_members(backend, line):
    proposal = _with_body(backend, line)
    assert validate_candidate(proposal, backend, 1, 1)["source"] == proposal["source"]


# ---- Imports and helper functions ----

LIBDEVICE_KERNEL = """{imports}

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    tl.store(out0 + offsets, libdevice.exp(values), offsets < N)
"""


def _triton(source):
    return dict(candidate(), source=source)


@pytest.mark.parametrize(
    "imports",
    [
        "import triton\nimport triton.language as tl\n"
        "from triton.language.extra.cuda import libdevice",
        "import triton\nimport triton.language as tl\n"
        "from triton.language.extra.cuda import libdevice as libdevice",
        "import triton\nimport triton.language as tl\n"
        "from triton.language.extra import libdevice",
        "import triton\nfrom triton import language as tl\n"
        "import triton.language.extra.libdevice as libdevice",
    ],
)
def test_triton_accepts_every_spelling_of_its_modules(imports):
    source = LIBDEVICE_KERNEL.format(imports=imports)
    assert validate_candidate(_triton(source), "triton", 1, 1)["source"] == source


@pytest.mark.parametrize(
    "imports, offending",
    [
        ("from triton.language.extra.cuda import exp", "from triton.language"),
        ("from triton.language.extra.cuda import libdevice as ld", "as ld"),
        ("from . import libdevice", "from . import libdevice"),
        ("import os", "import os"),
        ("from triton.runtime import driver", "from triton.runtime"),
    ],
)
def test_other_imports_are_named_with_their_line(imports, offending):
    source = LIBDEVICE_KERNEL.format(
        imports="import triton\nimport triton.language as tl\n" + imports
    )
    with pytest.raises(ValueError) as error:
        validate_candidate(_triton(source), "triton", 1, 1)
    message = str(error.value)
    assert message.startswith("line 3: ") and offending in message
    assert "`from triton.language.extra.cuda import libdevice`" in message


def test_module_level_statements_are_named_with_their_line():
    source = TRITON_SOURCE.replace(
        "import triton.language as tl\n", "import triton.language as tl\nBLOCK = 128\n"
    )
    with pytest.raises(ValueError, match=r"line 3: 'BLOCK = 128' is not allowed"):
        validate_candidate(_triton(source), "triton", 1, 1)


TRITON_HELPERS = """import triton
import triton.language as tl

@triton.jit
def _double(x, SCALE: tl.constexpr):
    return x * SCALE

@triton.jit
def _add(a, b):
    return a + b

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    total = tl.reduce(_double(values, 2), 0, _add)
    tl.store(out0 + offsets, values + total, offsets < N)
"""


def test_triton_accepts_jit_helpers_called_or_passed_by_name():
    source = TRITON_HELPERS
    assert validate_candidate(_triton(source), "triton", 1, 1)["source"] == source


def test_helper_parameters_may_be_named_like_other_builtins():
    source = TRITON_HELPERS.replace(
        "def _add(a, b):\n    return a + b",
        "def _add(input, id):\n    return input + id",
    )
    assert source != TRITON_HELPERS
    assert validate_candidate(_triton(source), "triton", 1, 1)["source"] == source


@pytest.mark.parametrize(
    "change, match",
    [
        (("@triton.jit\ndef _double", "def _double"), "must be decorated @triton.jit"),
        (("@triton.jit\ndef _double", "@triton.jit()\ndef _double"), "decorated"),
        (("def _double(x, SCALE", "def N(x, SCALE"), "needs a name other than"),
        (("def _double(x, SCALE", "def arg0(x, SCALE"), "needs a name other than"),
        (("def _double(x, SCALE", "def min(x, SCALE"), "needs a name other than"),
        (("x, SCALE: tl.constexpr)", "x, SCALE: tl.constexpr = 2)"), "defaults"),
        (("x, SCALE: tl.constexpr)", "x, SCALE: int)"), "annotated tl.constexpr"),
        (("    return x * SCALE", "    import os\n    return x"), "device computation"),
        (("    return x * SCALE", "    return open(x)"), "'open' is not allowed"),
        (("    return x * SCALE", "    return x.__class__"), "private Python"),
        (
            ("    return a + b", "    _double = a\n    return a"),
            "cannot bind '_double'",
        ),
        (("def _add(a, b):", "def _add(abs, b):"), "parameter 'abs' would rebind"),
        (("def _add(a, b):", "def _add(a, _double):"), "parameter '_double'"),
        (("def _add(a, b):", "def _add(a, tl):"), "parameter 'tl' would rebind"),
        (("    return a + b", "    return a + tl"), "DSL module 'tl' only"),
    ],
)
def test_helpers_follow_the_kernel_rules(change, match):
    source = TRITON_HELPERS.replace(*change)
    assert source != TRITON_HELPERS
    with pytest.raises(ValueError, match=match):
        validate_candidate(_triton(source), "triton", 1, 1)


def test_the_named_kernel_must_still_be_defined():
    proposal = dict(_triton(TRITON_HELPERS), kernel_name="other")
    with pytest.raises(ValueError, match=r"exactly the named kernel 'other'.*_add"):
        validate_candidate(proposal, "triton", 1, 1)


def test_a_function_defined_twice_is_rejected():
    source = TRITON_HELPERS + "\n@triton.jit\ndef _add(a, b):\n    return a - b\n"
    with pytest.raises(ValueError, match="'_add' is defined twice"):
        validate_candidate(_triton(source), "triton", 1, 1)


CUTILE_HELPERS = """{imports}

@ct.function
def scaled(tile, factor: ct.Constant[float]):
    return tile * factor

def shifted(tile):
    return tile + 1

@ct.kernel
def generated(arg0, out0, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    values = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,))
    ct.store(out0, index=(ct.bid(0),), tile=shifted(scaled(values, 2.0)))
"""


@pytest.mark.parametrize(
    "imports", ["import cuda.tile as ct", "from cuda import tile as ct"]
)
def test_cutile_accepts_function_and_plain_helpers(imports):
    source = CUTILE_HELPERS.format(imports=imports)
    proposal = dict(candidate("cutile"), source=source)
    assert validate_candidate(proposal, "cutile", 1, 1)["source"] == source
    with pytest.raises(ValueError, match="must be decorated @ct.function or"):
        validate_candidate(
            dict(proposal, source=source.replace("@ct.function", "@ct.kernel")),
            "cutile",
            1,
            1,
        )
