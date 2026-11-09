# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Saved references are data: loading a run never executes code from it."""

import json

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torch_tensorrt.dynamo.region_compilation._generation import _reference

# Every str argument of an operation that a reference may call. A new name can
# be a file or other resource: review the operation, and refuse it in
# _reference._UNSAFE if it has effects, before adding the name here.
_STRING_ARGUMENTS = {
    "UPLO",
    "a",
    "activation",
    "algorithm",
    "approximate",
    "assert_msg",
    "b",
    "driver",
    "equation",
    "indexing",
    "interpolation",
    "mode",
    "norm",
    "ord",
    "p",
    "pad_mode",
    "padding",
    "padding_side",
    "reduce",
    "rounding_mode",
    "side",
}


class _Mixed(torch.nn.Module):
    def forward(self, x, y):
        z = torch.ops.aten._to_copy.default(x, dtype=torch.float16)
        values, indices = torch.ops.aten.max.dim(y, -1, True)
        return (
            torch.ops.aten.clamp.default(z.float(), float("-inf"), 1e3) + values,
            indices,
            torch.ops.aten.reshape.default(y, [2, -1]),
            torch.ops.aten.__lshift__.Scalar(indices, 1),
        )


def _reference_data(*nodes, constants=None):
    """A one-input reference whose output is the last node."""
    entries = [
        {"name": "arg0", "op": "placeholder", "args": {"tuple": []}, "kwargs": {}},
        *nodes,
    ]
    output = {"tuple": [{"node": entries[-1]["name"]}]}
    entries.append({"name": "output", "op": "output", "args": output, "kwargs": {}})
    return {"version": 1, "constants": constants or {}, "nodes": entries}


def _call(name, target, *args, **kwargs):
    return {
        "name": name,
        "op": "call_function",
        "target": target,
        "args": {"tuple": list(args)},
        "kwargs": kwargs,
    }


def test_reference_round_trips_as_json(tmp_path):
    inputs = (torch.randn(2, 6), torch.randn(2, 6))
    reference = make_fx(_Mixed())(*inputs)
    path = tmp_path / _reference.FILENAME
    _reference.save_reference(reference, path)
    json.loads(path.read_text())  # plain JSON, no pickle
    restored = _reference.load_reference(path)
    for actual, expected in zip(restored(*inputs), reference(*inputs)):
        torch.testing.assert_close(actual, expected)
    targets = [node.target for node in reference.graph.nodes]
    assert [node.target for node in restored.graph.nodes][2:] == targets[2:]
    # Names follow the controller's own references, whatever the file says.
    names = [node.name for node in restored.graph.nodes]
    values = [f"value{position + 1}" for position in range(2, len(names) - 1)]
    assert names == ["arg0", "arg1", *values, "output"]


def test_non_aten_targets_are_not_saved():
    reference = torch.fx.symbolic_trace(lambda x: torch.nn.functional.gelu(x))
    with pytest.raises(TypeError, match="cannot save call target"):
        _reference.encode_reference(reference)


@pytest.mark.parametrize(
    "target",
    ["builtins::eval", "os::system", "torch::load", "tensorrt::execute_engine.default"],
)
def test_loading_resolves_only_aten_prims_and_operator(target):
    data = _reference_data(_call("value1", target, "echo pwned"))
    with pytest.raises(ValueError, match="operation"):
        _reference.decode_reference(data)


def test_saved_names_never_reach_generated_code(tmp_path):
    marker = tmp_path / "ran"
    # FX writes a placeholder's name into the forward() it generates and runs,
    # where this one would be a default argument executed while loading.
    crafted = f"arg0=open({str(marker)!r}, 'w').write('ran')"
    data = _reference_data(
        _call("x = 1", "aten::relu.default", {"node": crafted}),
    )
    data["nodes"][0]["name"] = crafted
    restored = _reference.decode_reference(data)
    assert not marker.exists()
    assert "open" not in restored.code
    assert [node.name for node in restored.graph.nodes] == ["arg0", "value2", "output"]
    torch.testing.assert_close(
        restored(torch.tensor([-1.0, 2.0])), torch.tensor([0, 2.0])
    )
    assert not marker.exists()


@pytest.mark.parametrize(
    "target, args",
    [
        ("aten::from_file.default", ["victim", True, 4]),  # creates and maps a file
        ("aten::fill_.Scalar", [{"node": "arg0"}, 1]),  # mutates
        ("aten::rand.default", [{"list": [3]}]),  # draws random numbers
        # Inductor's random primitives carry no random tag
        ("prims::inductor_random.default", [{"list": [3]}, {"node": "arg0"}, "rand"]),
        ("prims::inductor_randint.default", [0, 9, {"list": [3]}, {"node": "arg0"}]),
        ("aten::_print.default", ["hi"]),  # runs only for its effect
        ("aten::save.default", [{"node": "arg0"}, "victim"]),  # TorchScript builtin
        ("aten::manual_seed.default", [0]),  # sets the global seed
        # A TorchScript builtin that writes into its first argument undeclared
        (
            "aten::_no_grad_embedding_renorm_.default",
            [{"node": "arg0"}] * 2 + [1.0, 2.0],
        ),
        ("aten::_copy_from.default", [{"node": "arg0"}, {"node": "arg0"}]),
        ("aten::__dict__.__setitem__", ["key", 1]),  # not operators
        ("aten::__and__.__init__", []),
        ("aten::add._schemas", []),
    ],
)
def test_operations_with_effects_are_not_loaded(target, args):
    import torch._inductor  # noqa: F401  registers the prims::inductor_* operators

    data = _reference_data(_call("value1", target, *args))
    with pytest.raises(ValueError, match="unsupported operation"):
        _reference.decode_reference(data)


def test_a_crafted_reference_cannot_write_a_file(tmp_path):
    victim, payload = tmp_path / "victim", b"owned"
    mapped = _call(
        "mapped",
        "aten::from_file.default",
        str(victim),
        True,
        len(payload),
        dtype={"dtype": "uint8"},
    )
    writes = [
        _call(f"byte{i}", "aten::fill_.Scalar", {"node": "mapped"}, byte)
        for i, byte in enumerate(payload)
    ]
    path = tmp_path / _reference.FILENAME
    path.write_text(json.dumps(_reference_data(mapped, *writes)))
    with pytest.raises(ValueError, match="unsupported operation"):
        _reference.load_reference(path)
    assert not victim.exists()


def test_saving_refuses_what_loading_refuses():
    def write(x):
        torch.ops.aten.fill_.Scalar(x, 1)
        return x

    with pytest.raises(TypeError, match="cannot save operation"):
        _reference.encode_reference(make_fx(write)(torch.zeros(3)))
    graph = torch.fx.Graph()
    path = graph.placeholder("path")
    graph.output(graph.call_function(torch.ops.aten.from_file.default, (path,)))
    with pytest.raises(TypeError, match="cannot save operation"):
        _reference.encode_reference(torch.fx.GraphModule(torch.nn.Module(), graph))


class _Seeded(torch.nn.Module):
    """Operations that PyTorch tags as random, called so that they draw none."""

    def forward(self, q, k, v, x, h, c, w_ih, w_hh, b_ih, b_hh):
        aten = torch.ops.aten
        noise = aten.zeros_like.default(x)
        lstm = aten.lstm.input(
            x, (h, c), [w_ih, w_hh, b_ih, b_hh], True, 1, 0.5, False, False, True
        )
        return (
            aten.scaled_dot_product_attention.default(q, k, v, is_causal=True),
            aten.scaled_dot_product_attention.default(q, k, v, None, 0.0),
            aten.dropout.default(x, 0.5, False),
            aten.rrelu_with_noise_functional.default(x, noise, 0.1, 0.3, False)[0],
            lstm[0],
        )


def test_attention_dropout_and_rnns_that_draw_no_random_numbers_round_trip(tmp_path):
    reference = torch.fx.symbolic_trace(_Seeded())
    path = tmp_path / _reference.FILENAME
    _reference.save_reference(reference, path)
    restored = _reference.load_reference(path)
    targets = {node.target for node in restored.graph.nodes}
    assert torch.ops.aten.scaled_dot_product_attention.default in targets
    assert torch.ops.aten.lstm.input in targets
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(8, 8, batch_first=True)
    inputs = (
        *(torch.randn(1, 2, 16, 8) for _ in range(3)),
        torch.randn(2, 5, 8),
        torch.zeros(1, 2, 8),
        torch.zeros(1, 2, 8),
        *(weight.detach() for weight in lstm.parameters()),
    )
    for actual, expected in zip(restored(*inputs), reference(*inputs)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "target, tensors, drawing, quiet",
    [
        (
            torch.ops.aten.scaled_dot_product_attention.default,
            3,
            ((), {"dropout_p": 0.5}),
            ((), {"dropout_p": 0.0}),
        ),
        (torch.ops.aten.dropout.default, 1, ((0.5, True), {}), ((0.5, False), {})),
        (
            torch.ops.aten.rrelu.default,
            1,
            ((0.1, 0.3, True), {}),
            ((0.1, 0.3, False), {}),
        ),
    ],
)
def test_calls_that_draw_random_numbers_are_neither_saved_nor_loaded(
    target, tensors, drawing, quiet
):
    def reference(args, kwargs):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(target, (x,) * tensors + args, kwargs))
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    with pytest.raises(TypeError, match="draws random numbers"):
        _reference.encode_reference(reference(*drawing))
    _reference.decode_reference(_reference.encode_reference(reference(*quiet)))
    # A crafted file cannot turn the random numbers back on.
    name = f"{target.namespace}::{target._opname}.{target._overloadname}"
    args, kwargs = drawing
    data = _reference_data(
        _call("value1", name, *[{"node": "arg0"}] * tensors, *args, **kwargs)
    )
    with pytest.raises(ValueError, match="draw random numbers"):
        _reference.decode_reference(data)


def test_fused_attention_operators_are_not_loaded():
    # PyTorch tags them as random, and no argument tells that they draw none.
    data = _reference_data(
        _call(
            "value1",
            "aten::_scaled_dot_product_flash_attention.default",
            *[{"node": "arg0"}] * 3,
            0.0,
        )
    )
    with pytest.raises(ValueError, match="unsupported operation"):
        _reference.decode_reference(data)


def test_loadable_operations_take_no_file_names():
    names = set()
    for schema in torch._C._jit_get_all_schemas():
        namespace, _, name = schema.name.partition("::")
        if namespace not in ("aten", "prims"):
            continue
        overloads = getattr(getattr(torch.ops, namespace), name)
        if _reference._supported(getattr(overloads, schema.overload_name or "default")):
            names |= {a.name for a in schema.arguments if "str" in str(a.type)}
    assert names <= _STRING_ARGUMENTS, sorted(names - _STRING_ARGUMENTS)


@pytest.mark.parametrize("target", ["training", "__class__", "forward", "missing"])
def test_only_saved_constants_are_read(target):
    data = _reference_data(
        {
            "name": "value1",
            "op": "get_attr",
            "target": target,
            "args": {"tuple": []},
            "kwargs": {},
        },
        constants={"defined": 1},
    )
    with pytest.raises(ValueError, match="does not define"):
        _reference.decode_reference(data)


@pytest.mark.parametrize("name", ["recompile", "graph", "class", "__foo"])
def test_constants_cannot_replace_module_attributes(name):
    data = _reference_data(
        {
            "name": "value1",
            "op": "get_attr",
            "target": name,
            "args": {"tuple": []},
            "kwargs": {},
        },
        constants={name: 1},
    )
    with pytest.raises(ValueError, match="invalid name"):
        _reference.decode_reference(data)


@pytest.mark.parametrize(
    "target, key",
    [
        ("aten::add.Tensor", "alpha2"),  # not an argument of add
        ("aten::add.Tensor", "other = arg0 if open('victim', 'w') else arg0)#"),
        # An argument name, but a Python keyword
        ("aten::_has_compatible_shallow_copy_type.default", "from"),
    ],
)
def test_keywords_must_name_operation_arguments(target, key):
    data = _reference_data(_call("value1", target, {"node": "arg0"}, **{key: 1}))
    with pytest.raises(ValueError, match="unexpected arguments"):
        _reference.decode_reference(data)


@pytest.mark.parametrize(
    "target, args, kwargs",
    [
        # A list or string operand could grow into an arbitrarily large value.
        ("operator::mul", [{"list": [0]}, 1 << 40], {}),
        ("operator::mul", ["x", 1 << 40], {}),
        ("operator::add", [{"node": "arg0"}], {}),
        ("operator::neg", [{"node": "arg0"}], {"x": 1}),
    ],
)
def test_python_operators_take_only_values_and_numbers(target, args, kwargs):
    data = _reference_data(_call("value1", target, *args, **kwargs))
    with pytest.raises(ValueError, match="unexpected arguments"):
        _reference.decode_reference(data)


def _nested(depth):
    value = 1
    for _ in range(depth):
        value = {"list": [value]}
    return value


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda data: data.update(version=0), "format version"),
        (
            lambda data: data["constants"].update({"__class__": 1}),
            "invalid name",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append({"dtype": "load"}),
            "unknown dtype",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append({"dtype": ["x"]}),
            "unknown dtype",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append({"device": "gpu:x"}),
            "unknown device",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append({"float": "1e5"}),
            "unexpected form",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append([1, 2]),
            "unexpected form",
        ),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append(_nested(100)),
            "nested too deeply",
        ),
        (lambda data: data["nodes"][1].update(op="call_module"), "unsupported node"),
        (
            lambda data: data["nodes"][1]["args"]["tuple"].append({"node": "nope"}),
            "undefined value",
        ),
        (
            lambda data: data["nodes"][1].update(name=data["nodes"][0]["name"]),
            "repeated name",
        ),
        (lambda data: data["nodes"][0].update(name=["arg0"]), "invalid or repeated"),
        (
            lambda data: data["nodes"].append(data["nodes"][0]),
            "end with its only output",
        ),
        (lambda data: data["nodes"].pop(), "end with its only output"),
    ],
)
def test_malformed_references_are_rejected(change, match):
    data = _reference.encode_reference(make_fx(torch.relu)(torch.randn(3)))
    change(data)
    with pytest.raises(ValueError, match=match):
        _reference.decode_reference(data)


def test_dtype_names_never_reach_torch_lazy_imports(monkeypatch):
    # torch's module __getattr__ imports submodules such as _inductor by name.
    data = _reference.encode_reference(make_fx(torch.relu)(torch.randn(3)))
    data["nodes"][1]["args"]["tuple"].append({"dtype": "float128"})
    requested, lazy = [], torch.__getattr__

    def spy(name):
        requested.append(name)
        return lazy(name)

    monkeypatch.setattr(torch, "__getattr__", spy)
    with pytest.raises(ValueError, match="unknown dtype"):
        _reference.decode_reference(data)
    assert "float128" not in requested


def test_oversized_or_deeply_nested_files_are_not_parsed(tmp_path, monkeypatch):
    path = tmp_path / _reference.FILENAME
    path.write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(ValueError, match="nested too deeply"):
        _reference.load_reference(path)
    _reference.save_reference(make_fx(torch.relu)(torch.randn(3)), path)
    monkeypatch.setattr(_reference, "_MAX_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="larger than"):
        _reference.load_reference(path)


def test_a_pickle_named_like_a_reference_is_not_loaded(tmp_path):
    class Payload:
        def __reduce__(self):
            return (open, (str(tmp_path / "pwned"), "w"))

    path = tmp_path / _reference.FILENAME
    torch.save(Payload(), path)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        _reference.load_reference(path)
    assert not (tmp_path / "pwned").exists()
