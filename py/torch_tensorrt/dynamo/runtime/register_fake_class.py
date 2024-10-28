import base64
from typing import Any

import torch


@torch.library.register_fake("tensorrt::execute_engine")
def execute_engine(inputs, trt_engine):
    breakpoint()
    return trt_engine(inputs)


# namespace::class_name
@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self) -> None:
        pass

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        engine_info = [info[1] for info in flattened_tq]
        engine_info[3] = base64.b64decode(engine_info[3])  # decode engine
        engine_info[4] = str(engine_info[4][0])  # input names
        engine_info[5] = str(engine_info[5][0])  # output names
        engine_info[6] = str(int(engine_info[6]))  # hw compatible
        trt_engine = torch.classes.tensorrt.Engine(engine_info)
        return trt_engine
