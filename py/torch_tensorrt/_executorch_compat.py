"""Compatibility for the deprecated torch_tensorrt.load(format='executorch') API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence, Union, cast


class Program:
    """The legacy run/forward interface over an installed ExecuTorch Module."""

    def __init__(self, module: Any, data: bytes) -> None:
        # ExecuTorch's buffer loader can reference these bytes without copying them.
        self._data = data
        self._module = module

    @property
    def method_names(self) -> Collection[str]:
        return cast(Collection[str], self._module.method_names())

    def run(self, inputs: Sequence[Any], method: str = "forward") -> Sequence[Any]:
        """Run a method, preserving the legacy CUDA-to-CPU input conversion."""
        import torch

        inputs = tuple(
            value.cpu() if isinstance(value, torch.Tensor) and value.is_cuda else value
            for value in inputs
        )
        if method not in self.method_names:
            raise ValueError(
                f"Unknown method {method!r}; available methods: {sorted(self.method_names)}"
            )
        return cast(Sequence[Any], self._module.run_method(method, inputs))

    def forward(self, *inputs: Any) -> Sequence[Any]:
        return self.run(inputs, "forward")


def load(path: Union[str, Path]) -> Program:
    """Load a program with embedded weights through ExecuTorch's Module API."""
    try:
        from torch_tensorrt_executorch_runtime import register
    except ModuleNotFoundError as error:
        if error.name != "torch_tensorrt_executorch_runtime":
            raise
        raise ImportError(
            "Loading an ExecuTorch program requires the Torch-TensorRT delegate "
            "(torch_tensorrt_executorch_runtime). Install the delegate and ExecuTorch "
            "from the same release matrix."
        ) from error

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ExecuTorch model not found: {model_path}")
    data = model_path.read_bytes()
    register()
    # The Module API honors device-tagged arenas; the host Program loader does not.
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    return Program(_load_for_executorch_from_buffer(data), data)
