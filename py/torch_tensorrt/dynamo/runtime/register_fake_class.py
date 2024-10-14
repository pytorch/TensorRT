from typing import Any

import torch


# namespace::class_name
@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self) -> None:
        pass

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        return cls(**dict(flattened_tq))
