# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise example input/output checks without importing the GPU compiler stack."""

import argparse
import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[4]
_EXPORT = _ROOT / "examples/torchtrt_executorch_example/export_device_resident.py"


def _export(
    monkeypatch, path, remove_guard=False, delegates=None, operators=(), copy_ops=None
):
    tree = ast.parse(_EXPORT.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    if remove_guard:
        guards = [node for node in main.body if isinstance(node, ast.If)]
        assert len(guards) == 1
        main.body.remove(guards[0])
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
        # Default to the real names, so a program carrying a boundary copy is rejected. An empty
        # tuple made the check vacuous: nothing could ever match it.
        "BOUNDARY_COPY_OPS": (
            ("_h2d_copy", "_d2h_copy") if copy_ops is None else copy_ops
        ),
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
                        inputs=[],
                        outputs=[],
                    )
                ]
            )
        ),
    }
    monkeypatch.setattr(sys, "argv", [str(_EXPORT), "--model_path", str(path)])
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(_EXPORT), "exec"),
        namespace,
    )
    namespace["main"]()


@pytest.mark.parametrize("existing", [False, True])
def test_export_rejects_reference_collision_before_writing(
    monkeypatch, tmp_path, existing
):
    path = tmp_path / "model.expected"
    if existing:
        path.write_bytes(b"keep existing output")
    with pytest.raises(SystemExit) as error:
        _export(monkeypatch, path)
    assert error.value.code == 2
    assert (
        path.read_bytes() == b"keep existing output" if existing else not path.exists()
    )


def test_export_preserves_separate_model_and_reference(monkeypatch, tmp_path):
    path = tmp_path / "model.pte"
    _export(monkeypatch, path)
    assert path.read_bytes() == b"exported program"
    assert path.with_suffix(".expected").read_text() == "[64,64]\n0.5000\n"


def test_collision_guard_removal_overwrites_model(monkeypatch, tmp_path):
    path = tmp_path / "model.expected"
    _export(monkeypatch, path, remove_guard=True)
    assert path.read_text() == "[64,64]\n0.5000\n"


@pytest.mark.parametrize("optimize", [0, 2])
@pytest.mark.parametrize(
    "is_cuda,remove_guard", [(False, False), (True, False), (False, True)]
)
def test_device_input_check_precedes_load(monkeypatch, optimize, is_cuda, remove_guard):
    path = _ROOT / "examples/executorch_reference_runner/load_model_device_resident.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if remove_guard:
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "not x.is_cuda"
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
            cuda=SimpleNamespace(is_available=lambda: True),
            # Honour the device the script asks for, rather than answering from the parameter
            # alone. A stub that ignores it reports a CUDA tensor even when the script forgot to
            # request one, so the guard below would pass while the example was broken.
            ones=lambda *args, device=None, **kwargs: SimpleNamespace(
                is_cuda=is_cuda and str(device) == "cuda"
            ),
        ),
        "_load_for_executorch": load,
    }
    monkeypatch.setattr(sys, "argv", [str(path), "--model_path", "unused.pte"])
    reaches_load = is_cuda or remove_guard
    error = LookupError if reaches_load else RuntimeError
    message = "reached native load" if reaches_load else "requires a CUDA input"
    with pytest.raises(error, match=message):
        exec(compile(tree, str(path), "exec", optimize=optimize), namespace)
    assert calls == (["unused.pte"] if reaches_load else [])


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
