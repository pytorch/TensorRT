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


def _export(monkeypatch, path, remove_guard=False):
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
        "BOUNDARY_COPY_OPS": (),
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
                            for name in ("TensorRTBackend", "CudaBackend")
                        ],
                        operators=[],
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
