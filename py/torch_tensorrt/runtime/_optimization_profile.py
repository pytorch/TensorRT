"""Runtime optimization-profile selection for multi-profile TensorRT engines.

Profile selection is **manual by default**: pin a profile by its integer index
for a ``with`` span. Pass ``"auto"`` to opt into shape-based auto-selection for
that span. State is saved on enter and restored on exit (stack semantics), so
nested ``with`` blocks compose.

Example::

    from torch_tensorrt.runtime import optimization_profile

    # profiles=[prefill, decode] -> index 1 is decode
    with optimization_profile(trt_gm, 1):
        out = trt_gm(inputs_embeds=embeds, past_key_values=kv)

    with optimization_profile(trt_gm, "auto"):
        out = trt_gm(x)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _collect_trt_modules(module: Any) -> List[Any]:
    """Return all TorchTensorRTModule instances reachable from ``module``."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    trt_modules: List[Any] = []
    if isinstance(module, TorchTensorRTModule):
        trt_modules.append(module)
    elif isinstance(module, torch.nn.Module):
        for submodule in module.modules():
            if isinstance(submodule, TorchTensorRTModule):
                trt_modules.append(submodule)
    return trt_modules


class _OptimizationProfileContext:
    """Context manager that pins (or auto-selects) an optimization profile.

    Applies the requested profile to every TensorRT submodule on enter, and
    restores each submodule's previous profile state on exit.
    """

    def __init__(self, module: Any, profile: Optional[Any]) -> None:
        self.module = module
        self.profile = profile
        self._saved_state: List[Tuple[Any, Optional[Tuple[Any, ...]]]] = []

    def __enter__(self) -> Any:
        trt_modules = _collect_trt_modules(self.module)
        if not trt_modules:
            logger.warning(
                "optimization_profile() found no TensorRT submodules to configure."
            )
        for trt_module in trt_modules:
            saved = trt_module.get_optimization_profile_state()
            self._saved_state.append((trt_module, saved))
            trt_module.set_optimization_profile(self.profile)
        return self.module

    def __exit__(self, *exc: Any) -> None:
        # restore_* re-applies all of (pinned, auto, active) captured on enter,
        # including the active TRT profile index switched inside the block.
        for trt_module, saved in reversed(self._saved_state):
            try:
                trt_module.restore_optimization_profile_state(saved)
            except Exception as e:  # pragma: no cover - defensive restore
                logger.warning(
                    f"Failed to restore optimization profile state on exit: {e}"
                )
        self._saved_state.clear()


def optimization_profile(
    module: Any, profile: Optional[Any]
) -> _OptimizationProfileContext:
    """Select the active TensorRT optimization profile for a ``with`` span.

    Args:
        module: A compiled ``GraphModule`` (or ``TorchTensorRTModule``) containing
                one or more TensorRT engines.
            profile: Profile index (``int``), the string ``"auto"`` to enable
                shape-based auto-selection, or ``None`` to clear.

    Returns:
        A context manager. Pinned/auto state is restored on exit.
    """
    return _OptimizationProfileContext(module, profile)
