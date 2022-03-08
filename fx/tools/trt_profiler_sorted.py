import tensorrt as trt
import operator


class SortedTRTProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.layers = {}

    def report_layer_time(self, layer_name: str, ms: int) -> None:
        self.layers[layer_name] = ms

    def print_sorted_profile(self) -> None:
        for k, v in sorted(self.layers.items(), key=operator.itemgetter(1)):
            print(f"{k}: {v}ms")


