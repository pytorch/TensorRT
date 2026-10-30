# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import io

import pytest
import torch
from torch.export.graph_signature import ConstantArgument, InputKind, TensorArgument
from torch_tensorrt.dynamo._exporter import create_trt_exp_program

CONSTANTS = [(2, 3), (2.0, 3.0), (True, False), ("fixed", "different"), (None, 0)]


@pytest.mark.unit
@pytest.mark.parametrize("value,different", CONSTANTS)
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_constant_input_signature_survives_save_load(
    value, different, keyword, dynamic
):
    """Unused frozen inputs must remain constants even when no examples are supplied."""

    class Model(torch.nn.Module):
        def forward(self, x, constant):
            return x + 1

    x = torch.randn(2, 4)
    args = (x,) if keyword else (x, value)
    kwargs = {"constant": value} if keyword else {}
    shapes = (
        {"x": {0: torch.export.Dim("batch", min=1, max=8)}, "constant": None}
        if dynamic
        else None
    )
    exported = torch.export.export(Model(), args, kwargs, dynamic_shapes=shapes)
    rebuilt = create_trt_exp_program(exported.module())
    user_specs = [
        spec
        for spec in rebuilt.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]
    assert isinstance(user_specs[0].arg, TensorArgument)
    assert isinstance(user_specs[1].arg, ConstantArgument)
    assert type(user_specs[1].arg.value) is type(value)
    assert user_specs[1].arg.value == value

    saved = io.BytesIO()
    torch.export.save(rebuilt, saved)
    saved.seek(0)
    loaded = torch.export.load(saved).module()
    for size in ((2, 5) if dynamic else (2,)):
        inputs = torch.randn(size, 4)
        result = loaded(inputs, constant=value) if keyword else loaded(inputs, value)
        torch.testing.assert_close(result, inputs + 1)

    # Match PyTorch's own input guards, including version-dependent handling of
    # unused None inputs, rather than adding guards during TensorRT export.
    native = exported.module()
    try:
        if keyword:
            expected = native(x, constant=different)
        else:
            expected = native(x, different)
    except (RuntimeError, ValueError):
        with pytest.raises((RuntimeError, ValueError)):
            if keyword:
                loaded(x, constant=different)
            else:
                loaded(x, different)
    else:
        actual = loaded(x, constant=different) if keyword else loaded(x, different)
        torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_constants_do_not_displace_lifted_parameters_or_buffers():
    """A constant preceding the tensor input must not precede lifted state."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.register_buffer("bias", torch.randn(4))

        def forward(self, scale, x, use_bias=True, mask=None):
            result = x @ self.weight * scale
            return result + self.bias if use_bias else result

    model = Model().eval()
    x = torch.randn(2, 4)
    exported = torch.export.export(model, (2.0, x), {"use_bias": True, "mask": None})
    rebuilt = create_trt_exp_program(exported.module())
    specs = rebuilt.graph_signature.input_specs
    placeholders = [node for node in rebuilt.graph.nodes if node.op == "placeholder"]
    assert [spec.arg.name for spec in specs] == [node.name for node in placeholders]
    assert [spec.kind for spec in specs[:2]] == [InputKind.PARAMETER, InputKind.BUFFER]
    assert [type(spec.arg) for spec in specs[2:]] == [
        ConstantArgument,
        TensorArgument,
        ConstantArgument,
        ConstantArgument,
    ]

    saved = io.BytesIO()
    torch.export.save(rebuilt, saved)
    saved.seek(0)
    loaded = torch.export.load(saved).module()
    torch.testing.assert_close(
        loaded(2.0, x, use_bias=True, mask=None),
        model(2.0, x, use_bias=True, mask=None),
    )


@pytest.mark.unit
def test_zero_dimensional_tensor_input_is_not_a_constant():
    class Model(torch.nn.Module):
        def forward(self, x, scale):
            return x * scale

    x, scale = torch.randn(2, 4), torch.tensor(2.0)
    exported = torch.export.export(Model(), (x, scale))
    rebuilt = create_trt_exp_program(exported.module())
    assert all(
        isinstance(spec.arg, TensorArgument)
        for spec in rebuilt.graph_signature.input_specs
    )
    saved = io.BytesIO()
    torch.export.save(rebuilt, saved)
    saved.seek(0)
    loaded = torch.export.load(saved).module()
    torch.testing.assert_close(loaded(x, torch.tensor(3.0)), x * 3)
