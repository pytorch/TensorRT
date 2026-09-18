# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise example input/output checks without importing the GPU compiler stack."""

import argparse
import ast
import enum
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[4]
_EXPORT = _ROOT / "examples/torchtrt_executorch_example/export_device_resident.py"


class _Tensor:
    """Stands in for the schema's tensor, which the device check tests with isinstance."""

    def __init__(self, extra_tensor_info=None):
        self.extra_tensor_info = extra_tensor_info


class _DeviceType(enum.IntEnum):
    CPU = 0
    CUDA = 1


def _export(
    monkeypatch,
    path,
    delegates=None,
    operators=(),
    boundary_devices=(),
):
    tree = ast.parse(_EXPORT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    tensor = SimpleNamespace(
        shape=(64, 64),
        cuda=lambda: tensor,
        flatten=lambda: [SimpleNamespace(item=lambda: 0.5)],
    )

    class Model:
        def eval(self):
            return self

        def cuda(self):
            return self

        def __call__(self, inputs):
            return tensor

    def save(model, filename, **kwargs):
        Path(filename).write_bytes(b"exported program")

    def config(**kwargs):
        return kwargs

    namespace = {
        "argparse": argparse,
        "Path": Path,
        "sys": sys,
        "CoalescedModel": Model,
        "SHAPE": (64, 64),
        # No BOUNDARY_COPY_OPS here on purpose. Supplying it meant the script's own constant was
        # never read, so emptying that constant, which is the exact failure the comment beside it
        # warns about, changed nothing. The module-level assignments run below instead.
        "torch": SimpleNamespace(
            no_grad=nullcontext,
            randn=lambda _: tensor,
            ones=lambda _: tensor,
            export=SimpleNamespace(export=lambda *args: None),
        ),
        "torch_tensorrt": SimpleNamespace(
            save=save, dynamo=SimpleNamespace(compile=lambda *args, **kwargs: None)
        ),
        "CudaPartitioner": lambda *args: None,
        "CudaBackend": SimpleNamespace(
            generate_method_name_compile_spec=lambda _: None
        ),
        # The two schema names the device check needs. A real enum, so the example's own
        # DeviceType(device).name works unchanged.
        "Tensor": _Tensor,
        "DeviceType": _DeviceType,
        "ExecutorchBackendConfig": config,
        "PropagateDeviceConfig": config,
        "MemoryPlanningPass": config,
        "deserialize_pte_binary": lambda _: SimpleNamespace(
            program=SimpleNamespace(
                execution_plan=[
                    SimpleNamespace(
                        delegates=[
                            SimpleNamespace(id=name)
                            for name in (
                                ("TensorRTBackend", "CudaBackend")
                                if delegates is None
                                else delegates
                            )
                        ],
                        operators=[
                            SimpleNamespace(name=n, overload="") for n in operators
                        ],
                        # A boundary carrying tensors, so the device check has something to
                        # inspect. Empty lists meant the whole check could be deleted unnoticed.
                        inputs=list(range(len(boundary_devices))),
                        outputs=[],
                        values=[
                            SimpleNamespace(
                                val=_Tensor(
                                    extra_tensor_info=(
                                        None
                                        if device is None
                                        else SimpleNamespace(device_type=device)
                                    )
                                )
                            )
                            for device in boundary_devices
                        ],
                    )
                ]
            )
        ),
    }
    monkeypatch.setattr(sys, "argv", [str(_EXPORT), "--model_path", str(path)])
    # The module's own assignments, then main. Compiling main alone left every module-level constant
    # to be supplied by this test, which is how the boundary names stopped being checked at all.
    module = ast.parse(_EXPORT.read_text(encoding="utf-8"))
    # The module's plain constant assignments, then main. Anything else at module level needs stubs
    # this test has no reason to grow, and the constants are what was going unread.
    body = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and all(isinstance(target, ast.Name) for target in node.targets)
        and isinstance(node.value, (ast.Tuple, ast.List, ast.Constant))
    ]
    body.append(main)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(_EXPORT), "exec"),
        namespace,
    )
    namespace["main"]()


# m.v2 is the case that decides the naming form: with_suffix() would replace .v2 and write
# m.expected, which is neither beside the program nor where a reader appending .expected looks.
# model.expected is here because appending is also what keeps the reference off the program.
@pytest.mark.parametrize("name", ["model.pte", "m.v2", "model.expected"])
def test_export_names_the_reference_by_appending_to_the_model_path(
    monkeypatch, tmp_path, name
):
    path = tmp_path / name
    _export(monkeypatch, path)
    assert path.read_bytes() == b"exported program"
    assert path.with_name(path.name + ".expected").read_text() == "[64,64]\n0.5000\n"
    substituted = path.with_suffix(".expected")
    assert substituted == path or not substituted.exists()


@pytest.mark.parametrize("optimize", [0, 2])
@pytest.mark.parametrize(
    "is_cuda,remove_guard", [(False, False), (True, False), (False, True)]
)
def test_the_cuda_gate_precedes_the_load(monkeypatch, optimize, is_cuda, remove_guard):
    """The gate has to stop the program before it loads anything, and survive optimized Python.

    It used to be an assertion, which optimized Python strips, so the parameter covering that is not
    incidental. The guard removed here is the availability gate, not a check on the tensor: a check on
    the tensor cannot fire, because asking for the device has already failed by then.
    """
    path = _ROOT / "examples/executorch_reference_runner/load_model_device_resident.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if remove_guard:
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not torch.cuda.is_available()"
        ]
        assert len(guards) == 1
        tree.body.remove(guards[0])
    calls = []

    def load(path):
        calls.append(path)
        raise LookupError("reached native load")

    namespace = {
        "argparse": argparse,
        "Path": Path,
        "torch": SimpleNamespace(
            float32=object(),
            # The parameter drives availability, which is the thing the gate asks about.
            cuda=SimpleNamespace(is_available=lambda: is_cuda),
            # Honour the device the script asks for, so a script that forgot to request one is
            # visible here rather than reported as a device tensor.
            ones=lambda *args, device=None, **kwargs: SimpleNamespace(
                is_cuda=str(device) == "cuda"
            ),
        ),
        "Runtime": SimpleNamespace(get=lambda: SimpleNamespace(load_program=load)),
    }
    monkeypatch.setattr(sys, "argv", [str(path), "--model_path", "unused.pte"])
    reaches_load = is_cuda or remove_guard
    error = LookupError if reaches_load else RuntimeError
    message = "reached native load" if reaches_load else "cannot run"
    with pytest.raises(error, match=message):
        exec(compile(tree, str(path), "exec", optimize=optimize), namespace)
    # The public loader takes a path object, so compare as text rather than requiring a string.
    assert [str(c) for c in calls] == (["unused.pte"] if reaches_load else [])


@pytest.mark.parametrize(
    "delegates,expected",
    [
        (("CudaBackend",), "missing"),
        (("TensorRTBackend",), "missing"),
        ((), "missing"),
    ],
)
def test_export_rejects_a_program_that_is_not_coalesced(
    monkeypatch, tmp_path, delegates, expected
):
    """The coalescing check never ran, because the stub program always carried both delegates."""
    with pytest.raises(SystemExit, match=expected):
        _export(monkeypatch, tmp_path / "m.pte", delegates=delegates)


def test_export_rejects_a_program_that_still_copies_at_the_boundary(
    monkeypatch, tmp_path
):
    """The copy check never ran either: the operator table was empty and the names it looks for
    were an empty tuple, so nothing could match and the rejection path was unreachable.
    """
    with pytest.raises(SystemExit, match="still copies across the method boundary"):
        _export(
            monkeypatch,
            tmp_path / "m.pte",
            operators=("aten::_h2d_copy_default",),
        )


def test_a_rejected_export_leaves_the_previous_program_alone(monkeypatch, tmp_path):
    """Saving straight over the target let a rejected export destroy a good program.

    The reference file beside it then described something the program no longer was, which is worse
    than no output at all because it looks like a successful export.
    """
    model_path = tmp_path / "m.pte"
    model_path.write_bytes(b"the good program")
    with pytest.raises(SystemExit, match="missing"):
        _export(monkeypatch, model_path, delegates=("CudaBackend",))
    assert model_path.read_bytes() == b"the good program"
    assert not list(tmp_path.glob("*.staged")), "the staging file was left behind"


@pytest.mark.parametrize("output_on_cuda", [True, False])
def test_the_runner_rejects_an_output_that_came_back_on_the_host(
    monkeypatch, tmp_path, output_on_cuda
):
    """The check the whole example exists for had nothing reaching it.

    The other device test stops at the load, so the output guard never ran and removing it left the
    suite green. This one lets the load succeed and controls only where the output claims to live.
    """
    path = _ROOT / "examples/executorch_reference_runner/load_model_device_resident.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    output = SimpleNamespace(
        is_cuda=output_on_cuda, device="cuda:0" if output_on_cuda else "cpu"
    )
    # Shaped like the public Program: method_names is an attribute, and a method is loaded first
    # and then executed, rather than named on every call.
    program = SimpleNamespace(
        method_names=["forward"],
        load_method=lambda name: SimpleNamespace(execute=lambda inputs: [output]),
    )
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "torch": SimpleNamespace(
            float32=object(),
            cuda=SimpleNamespace(is_available=lambda: True),
            ones=lambda *args, device=None, **kwargs: SimpleNamespace(
                is_cuda=str(device) == "cuda"
            ),
        ),
        "Runtime": SimpleNamespace(
            get=lambda: SimpleNamespace(load_program=lambda _: program)
        ),
    }
    monkeypatch.setattr(sys, "argv", [str(path), "--model_path", "unused.pte"])
    if output_on_cuda:
        # It gets past the guard and fails later on something this stub does not provide. Anything
        # except the guard's own complaint proves the guard let a device output through.
        with pytest.raises(Exception) as caught:
            exec(compile(tree, str(path), "exec"), namespace)
        assert "output came back on" not in str(caught.value), caught.value
    else:
        with pytest.raises(AssertionError, match="output came back on"):
            exec(compile(tree, str(path), "exec"), namespace)


def test_the_runner_refuses_to_start_without_cuda(monkeypatch):
    """The gate that stops the runner on a machine with no CUDA had nothing exercising it.

    Every other case stubs CUDA as available, so deleting the gate left the whole suite green.
    """
    path = _ROOT / "examples/executorch_reference_runner/load_model_device_resident.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "torch": SimpleNamespace(
            float32=object(),
            cuda=SimpleNamespace(is_available=lambda: False),
            ones=lambda *args, device=None, **kwargs: SimpleNamespace(is_cuda=False),
        ),
        "Runtime": SimpleNamespace(
            get=lambda: SimpleNamespace(load_program=lambda _: None)
        ),
    }
    monkeypatch.setattr(sys, "argv", [str(path), "--model_path", "unused.pte"])
    # The specific message, not just any mention of CUDA. With this gate deleted the input guard
    # further down raises its own CUDA complaint, so a loose match passes either way and the check
    # says nothing about the gate it is named for.
    with pytest.raises(
        RuntimeError, match="cannot run\nwithout CUDA|cannot run without CUDA"
    ):
        exec(compile(tree, str(path), "exec"), namespace)


@pytest.mark.parametrize(
    "devices,rejected",
    [
        ((_DeviceType.CUDA,), False),
        ((_DeviceType.CUDA, _DeviceType.CUDA), False),
        ((_DeviceType.CPU,), True),
        ((_DeviceType.CUDA, _DeviceType.CPU), True),
        ((None,), True),
    ],
)
def test_export_rejects_a_boundary_tensor_that_is_not_on_the_device(
    monkeypatch, tmp_path, devices, rejected
):
    """Deleting the whole device check kept the suite green, because no boundary carried tensors.

    A program whose method boundary holds host memory defeats the point of a device-resident export,
    and the caller would find out at run time instead. A tensor with no device information counts as
    host, which is what the example's own default says.
    """
    path = tmp_path / "m.pte"
    path.write_bytes(b"program")
    if rejected:
        with pytest.raises(SystemExit) as raised:
            _export(monkeypatch, path, boundary_devices=devices)
        assert "non-CUDA method boundary tensors" in str(raised.value), raised.value
    else:
        _export(monkeypatch, path, boundary_devices=devices)
