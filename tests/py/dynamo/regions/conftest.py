# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend


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
