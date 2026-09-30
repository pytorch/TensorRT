"""Numerical checks with actual LeRobot PI0.5 layers at reduced dimensions."""

import os
import subprocess
import sys

import pytest
import torch
from torch_tensorrt_edge_llm.pi05 import (
    PI05Action,
    PI05Prefill,
    export_pi05,
    prepare_pi05_sample,
    write_native_inputs,
)


def inputs(core, device):
    images = [torch.randn(2, 3, 16, 16, device=device) for _ in range(2)]
    tokens = torch.randint(0, 60, (2, 3), device=device)
    image_masks = [
        torch.tensor([True, True], device=device),
        torch.tensor([True, False], device=device),
    ]
    token_mask = torch.tensor([[True, True, False], [True, True, True]], device=device)
    noise = torch.randn(2, 3, 4, device=device)
    sample = prepare_pi05_sample(core, images, image_masks, tokens, token_mask, noise)
    return sample, (images, image_masks, tokens, token_mask)


def test_pi05_adapters_match_reference_and_preserve_prefix():
    pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
    from pi05_fixture import tiny_core

    torch.manual_seed(7)
    core = tiny_core()
    sample, observation = inputs(core, "cpu")
    from torch_tensorrt_edge_llm import ops
    from torch_tensorrt_edge_llm.pi05 import PI05Vision

    with torch.no_grad():
        prefix = ops.fuse_prefix(
            PI05Vision(core, batch_size=2, cameras=2)(sample[0]), sample[1], sample[2]
        )
        _, k, v = PI05Prefill(core)(prefix, sample[3], sample[4])
        saved = k.clone(), v.clone()
        x = sample[5].clone()
        action = PI05Action(core)
        for step in range(3):
            x = (
                x
                - action(x, torch.full((2,), 1 - step / 3), k, v, sample[6], sample[7])
                / 3
            )
        reference = core.sample_actions(
            *observation, noise=sample[5].clone(), num_steps=3
        )
    torch.testing.assert_close(x, reference)
    torch.testing.assert_close(k, saved[0])
    torch.testing.assert_close(v, saved[1])


@pytest.mark.gpu
def test_pi05_compiled_export_fresh_process_and_native(tmp_path):
    pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
    pytest.importorskip("executorch.exir")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT integration requires CUDA")
    from executorch.exir._serialize._program import deserialize_pte_binary
    from exporters import EdgeExporter
    from pi05_fixture import tiny_core

    torch.manual_seed(7)
    core = tiny_core("cuda")
    sample, observation = inputs(core, "cuda")
    with torch.no_grad():
        reference = core.sample_actions(
            *observation, noise=sample[5].clone(), num_steps=3
        )
        exporter = EdgeExporter()
        exported = exporter.export_pi05(
            core,
            {
                "images": observation[0],
                "image_masks": observation[1],
                "tokens": observation[2],
                "token_mask": observation[3],
                "noise": sample[5],
            },
            {
                "engine_dir": tmp_path / "engines",
                "trt_settings": {"optimization_level": 0},
            },
            num_steps=3,
        )
        actual = exported.program.module()(*sample)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    ep = exported.save(tmp_path)
    pte = exported.save(tmp_path, output_format="executorch")
    program = deserialize_pte_binary(pte.read_bytes()).program
    assert {plan.name for plan in program.execution_plan} == {
        "vision",
        "prefill",
        "action_step",
    }
    assert all(
        [delegate.id for delegate in plan.delegates] == ["EdgeLLMBackend"]
        for plan in program.execution_plan
    )
    torch.save(tuple(t.detach().cpu() for t in sample), tmp_path / "inputs.pt")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,torch,torch_tensorrt_edge_llm; "
            "ep=torch.export.load(sys.argv[1]); "
            "args=tuple(t.cuda() for t in torch.load(sys.argv[2],weights_only=True)); "
            "torch.save(ep.module()(*args).cpu(),sys.argv[3])",
            str(ep),
            str(tmp_path / "inputs.pt"),
            str(tmp_path / "fresh.pt"),
        ],
        check=True,
        timeout=60,
        capture_output=True,
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "fresh.pt", weights_only=True),
        reference.cpu(),
        rtol=1e-4,
        atol=1e-5,
    )
    runner = os.environ.get("EDGELLM_PI05_RUNNER")
    if runner:
        write_native_inputs(sample, tmp_path / "native_inputs")
        subprocess.run(
            [
                runner,
                f"--model_path={pte}",
                f"--inputs_dir={tmp_path / 'native_inputs'}",
                f"--output_path={tmp_path / 'actions.bin'}",
                "--num_steps=3",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )
        native = torch.frombuffer(
            bytearray((tmp_path / "actions.bin").read_bytes()), dtype=torch.float32
        ).reshape_as(reference.cpu())
        torch.testing.assert_close(native, reference.cpu(), rtol=1e-4, atol=1e-5)


def test_export_rejects_wrong_camera_batch_before_compilation(tmp_path):
    from types import SimpleNamespace

    core = SimpleNamespace(config=SimpleNamespace())
    sample = (
        torch.zeros(1, 16, 16, 3),
        torch.zeros(2, 3, 4),
        *[torch.zeros(1) for _ in range(6)],
    )
    with pytest.raises(ValueError, match="camera-first"):
        export_pi05(core, sample, engine_dir=tmp_path, cameras=2)
