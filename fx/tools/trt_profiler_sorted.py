import json
import operator
from typing import Optional, Mapping, List

import tensorrt as trt
import torch
from fx2trt_oss.fx import TRTModule


class SortedTRTProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layers = {}

    def report_layer_time(self, layer_name: str, ms: int) -> None:
        self.layers[layer_name] = ms

    def print_sorted_profile(
        self, additional_info: Optional[Mapping[str, str]]
    ) -> None:
        additional_info = {} if additional_info is None else additional_info
        for k, v in sorted(self.layers.items(), key=operator.itemgetter(1)):
            additional_str = additional_info.get(k, "")
            print(f"{k} {additional_str}: {v}ms")


def profile_trt_module(
    name: str, trt_mod: TRTModule, mod_input: List[torch.Tensor]
) -> None:
    """
    Provide per layer timing and shape info
    """
    layer_info = json.loads(trt_mod.get_layer_info())  # pyre-ignore[29]
    shape_map = {}
    for layer in layer_info["Layers"]:
        name = layer["Name"]
        input_str = ", ".join(
            [str(x.get("Dimensions", "[]")) for x in layer.get("Inputs", [])]
        )
        output_str = ", ".join(
            [str(x.get("Dimensions", "[]")) for x in layer.get("Outputs", [])]
        )
        shape_map[name] = f"({input_str}) -> ({output_str})"

    trt_mod.enable_profiling(profiler=SortedTRTProfiler())  # pyre-ignore[29]
    _ = trt_mod(*mod_input)
    trt_mod.context.profiler.print_sorted_profile(shape_map)  # pyre-ignore[16]
    trt_mod.disable_profiling()  # pyre-ignore[29]
