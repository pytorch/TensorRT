from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EdgeConfig:
    """Knobs for :class:`~torch_tensorrt.hf.exporters.EdgeExporter`.

    ``strict`` / ``dynamic`` / ``dynamic_shapes`` match HuggingFace
    ``DynamoConfig`` so this can subclass it later without an API break.
    ``components`` is ``None`` to use the spec default (1 engine for an LLM,
    3–4 for a VLA).
    """

    strict: bool = False
    dynamic: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False

    engine_dir: Path | str | None = None
    max_seq_len: int = 968
    generation_reserve: int = 0
    components: tuple[str, ...] | None = None
    trt_settings: dict[str, Any] = field(default_factory=dict)
    dryrun: bool = False
    skip_runtime_export: bool = False
    model_type: str | None = None
