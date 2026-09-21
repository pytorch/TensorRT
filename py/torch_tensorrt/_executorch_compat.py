# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
    # The path first. A caller who mistyped a file name and also has no delegate installed was told
    # to install the delegate, which is true but is not what they got wrong.
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ExecuTorch model not found: {model_path}")
    try:
        import torch_tensorrt_executorch_runtime as delegate
    except ModuleNotFoundError as error:
        if error.name != "torch_tensorrt_executorch_runtime":
            raise
        raise ImportError(
            "Loading an ExecuTorch program requires the Torch-TensorRT delegate "
            "(torch_tensorrt_executorch_runtime). Install the delegate and ExecuTorch "
            "from the same release matrix."
        ) from error
    # A companion published before the delegate became a single registration call exposes
    # activate() instead. Accept it so upgrading this wheel alone keeps loading programs.
    register = getattr(delegate, "register", None)
    # Whether this companion predates single-call registration, which is the only thing the rewrite
    # below is about. Deciding that from the raised error's class name does not work: the older
    # companion defines a class of the same name, so a name test matches both and the rewrite it
    # guards never fires for the companion it exists for.
    predates_register = register is None
    if predates_register:
        register = getattr(delegate, "activate", None)
    if register is None:
        raise ImportError(
            "The installed torch_tensorrt_executorch_runtime exposes neither register() "
            "nor activate(). Install a delegate from the same release matrix as "
            "Torch-TensorRT."
        )

    data = model_path.read_bytes()
    try:
        register()
    except ImportError as error:
        # A current companion's own compatibility error already says precisely what is wrong, so
        # let it through rather than replacing it with a guess about the companion's age.
        if not predates_register:
            raise
        # A companion published before registration became a single call swapped in its own copy of
        # ExecuTorch's bindings, and refuses once ExecuTorch's own copy is already loaded. Its advice
        # is to import it earlier, which a caller of this function cannot do: the import it collides
        # with happens inside this library. Say the thing that does work instead.
        raise ImportError(
            "The installed torch_tensorrt_executorch_runtime is too old to register alongside "
            "ExecuTorch's own bindings. Upgrade it to a build that registers on import, from the "
            f"same release matrix as Torch-TensorRT. Underlying error: {error}"
        ) from error
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    # No eager validation here, deliberately. A truncated or altered program loads and fails only
    # when something first asks it a question, which reads badly, but this function exists to behave
    # exactly as the released one did and a test pins both the timing and the identity of that error.
    # Improving it means a new entry point, not a change to this one.
    return Program(_load_for_executorch_from_buffer(data), data)
