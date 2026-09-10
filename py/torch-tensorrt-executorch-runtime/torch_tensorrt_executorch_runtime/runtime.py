"""Python inference API for Torch-TensorRT ExecuTorch programs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence, Union, cast


def _load_module(data: bytes) -> Any:
    """Load a program through ExecuTorch's Module API.

    A TensorRT delegated program is exported with device-tagged memory-planned
    arenas, and ExecuTorch backs those with real device memory only through
    this API. Its program loader plans every arena on the host, so the device
    copy the exporter inserts around the delegate would hand ``cudaMemcpy`` a
    host destination and fail with ``invalid argument``.
    """
    try:
        from torch_tensorrt_executorch_runtime import activate, get_runtime
    except ImportError as error:
        raise ImportError(
            "ExecuTorch Python inference requires the prebuilt delegate. "
            'Install it with: pip install "torch-tensorrt[executorch]"'
        ) from error

    # get_runtime verifies TensorRTBackend is registered; activate returns the native module it
    # installed as the process portable runtime, which is what loads the program.
    get_runtime()
    native = activate()
    # Taken off the native module rather than imported through
    # executorch.extension.pybindings.portable_lib. That wrapper's presence in sys.modules is how
    # activate() detects that ExecuTorch's stock runtime was imported first, so importing it here
    # would make a later activate() in the same process refuse.
    return native._load_for_executorch_from_buffer(data)


class Program:
    """A loaded ExecuTorch program backed by TensorRTBackend.

    The ExecuTorch Python portable runtime executes across a CPU tensor
    boundary: CUDA inputs are copied to CPU before dispatch and outputs are
    returned on CPU. TensorRT still executes the delegated graph on GPU, but
    the device-resident input/output fast path is available only through the
    ExecuTorch C++ runner.
    """

    def __init__(self, module: Any, data: bytes) -> None:
        # ExecuTorch's BufferDataLoader references this memory without copying it.
        self._data = data
        self._module = module

    @property
    def method_names(self) -> Collection[str]:
        return cast(Collection[str], self._module.method_names())

    def run(self, inputs: Sequence[Any], method: str = "forward") -> Sequence[Any]:
        """Run a method using CPU inputs and return CPU outputs.

        CUDA tensor inputs are copied to CPU before entering the portable
        Python runtime. Use the C++ runner when inputs and outputs must remain
        device-resident.
        """
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
    """Load a `.pte` with the delegate-enabled ExecuTorch Python runtime.

    External `.ptd` weight files are not supported; weights must be embedded
    in the `.pte` file.
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ExecuTorch model not found: {model_path}")
    data = model_path.read_bytes()
    return Program(_load_module(data), data)


__all__ = ["Program", "load"]
