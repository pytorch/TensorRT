# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from filelock import FileLock
from torch._dynamo.testing import CompileCounterWithBackend

from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.region import KernelGenerator

from ._generation_utils import _passed, _proposal


@pytest.fixture(autouse=True)
def assume_qdp_plugins(monkeypatch):
    """Let generation checks pass on TensorRT builds without QDP plugins.

    There, as on TensorRT-RTX, tests that build or run a plugin skip, and the
    others need none. Tests of the QDP requirement clear the flag themselves.
    """
    monkeypatch.setattr(
        _factory,
        "ENABLED_FEATURES",
        _factory.ENABLED_FEATURES._replace(qdp_plugin=True),
    )


@pytest.fixture
def observed_inductor(monkeypatch):
    """Count real Inductor compilation and execution."""
    torch._dynamo.reset()
    real_compile = torch.compile
    state = SimpleNamespace(
        counter=CompileCounterWithBackend("inductor"),
        requests=[],
        executions=0,
        capture_states=[],
    )

    def compile_spy(model, **kwargs):
        state.requests.append(dict(kwargs))
        assert kwargs["backend"] == "inductor"
        assert kwargs["fullgraph"] is True
        assert kwargs["dynamic"] is False
        options = kwargs.pop("options")
        assert options == {"triton.cudagraphs": False}

        def backend(graph, inputs):
            state.capture_states.append(
                torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
            )
            with torch._inductor.config.patch(options):
                compiled = state.counter(graph, inputs)

            def run(*args):
                state.executions += 1
                return compiled(*args)

            return run

        kwargs["backend"] = backend
        return real_compile(model, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_spy)
    yield state
    torch._dynamo.reset()


@pytest.fixture
def factory(tmp_path, monkeypatch):
    """Run the generation controller with a scripted author and evaluator."""

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(x, x))

    boundary = SimpleNamespace(
        shape=(7,), dtype=torch.float32, device=torch.device("cuda:0")
    )
    monkeypatch.setattr(_factory, "_boundaries", lambda *args: ([boundary], [boundary]))
    monkeypatch.setattr(
        _factory, "_target", lambda *args: {"name": "test-gpu", "capability": [9, 0]}
    )
    author = Mock()
    evaluate = Mock()
    finalize = Mock(return_value=_passed(0.2))

    def propose(**kwargs):
        # Tests script bare candidates; a provider replies with the envelope.
        reply = author(**kwargs)
        return reply if "action" in reply else _proposal(reply)

    def dispatch_evaluation(request, directory, timeout):
        # Final validation and cache checks are holdout evaluations.
        holdout = request.get("validation_phase") == "holdout"
        return (finalize if holdout else evaluate)(request, directory, timeout)

    register = Mock(side_effect=lambda candidate, *args: candidate)
    monkeypatch.setattr(_factory._providers, "propose", propose)
    monkeypatch.setattr(_factory, "_evaluate", dispatch_evaluation)
    monkeypatch.setattr(_factory, "make_registration", register)
    # The real lock is keyed by the device UUID, which needs a GPU.
    monkeypatch.setattr(
        _factory, "gpu_lock", lambda root, device: FileLock(str(root / "gpu.lock"))
    )
    return SimpleNamespace(
        graph=torch.fx.symbolic_trace(Reference()),
        config=KernelGenerator(cache_dir=tmp_path / "cache", max_attempts=1),
        author=author,
        evaluate=evaluate,
        finalize=finalize,
        register=register,
        root=tmp_path / "cache",
    )
