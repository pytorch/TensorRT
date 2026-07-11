"""Python inference API for Torch-TensorRT ExecuTorch programs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union


def _runtime():
    try:
        from torch_tensorrt_executorch_delegate import runtime
    except ImportError as error:
        raise ImportError(
            "ExecuTorch Python inference requires the prebuilt delegate. "
            'Install it with: pip install "torch-tensorrt[executorch]"'
        ) from error
    return runtime()


class Program:
    """A loaded ExecuTorch program backed by TensorRTBackend."""

    def __init__(self, program: Any) -> None:
        self._program = program

    @property
    def method_names(self):
        return self._program.method_names

    def run(self, inputs: Sequence[Any], method: str = "forward") -> Sequence[Any]:
        import torch

        inputs = tuple(
            value.cpu() if isinstance(value, torch.Tensor) and value.is_cuda else value
            for value in inputs
        )
        if method not in self.method_names:
            raise ValueError(
                f"Unknown method {method!r}; available methods: {sorted(self.method_names)}"
            )
        loaded = self._program.load_method(method)
        if loaded is None:
            raise RuntimeError(f"ExecuTorch failed to load method {method!r}")
        return loaded.execute(inputs)

    def forward(self, *inputs: Any) -> Sequence[Any]:
        return self.run(inputs, "forward")


def load(path: Union[str, Path]) -> Program:
    """Load a `.pte` with the delegate-enabled ExecuTorch Python runtime."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ExecuTorch model not found: {model_path}")
    return Program(_runtime().load_program(model_path.read_bytes()))


__all__ = ["Program", "load"]
