from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
from torch_tensorrt.hf.exporters.spec import EdgeSpec


class EdgeRuntimeModule(nn.Module):  # type: ignore[misc]
    """Graph dumped by ``EdgeExporter``: packing in Python, compute as execute_engine."""

    def __init__(self, spec: EdgeSpec, engines: Mapping[str, str]) -> None:
        super().__init__()
        self.spec = spec
        self.engines = dict(engines)

    def forward(self, **sample: Any) -> Any:
        return self.spec.run(self.engines, sample)
