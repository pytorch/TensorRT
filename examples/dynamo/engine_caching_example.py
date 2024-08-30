import os
from typing import Optional

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_caching import BaseEngineCache

np.random.seed(0)
torch.manual_seed(0)

model = models.resnet18(pretrained=True).eval().to("cuda")
enabled_precisions = {torch.float}
debug = False
min_block_size = 1
use_python_runtime = False


def remove_timing_cache(path=TIMING_CACHE_PATH):
    if os.path.exists(path):
        os.remove(path)


def dynamo_compile(iterations=3):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
    # Mark the dim0 of inputs as dynamic
    batch = torch.export.Dim("batch", min=1, max=200)
    exp_program = torch.export.export(
        model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
    )

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand((100 + i, 3, 224, 224)).to("cuda")]
        remove_timing_cache()  # remove timing cache just for engine caching messurement
        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(inputs),
            use_python_runtime=use_python_runtime,
            enabled_precisions=enabled_precisions,
            debug=debug,
            min_block_size=min_block_size,
            make_refitable=True,
            cache_built_engines=cache_built_engines,
            reuse_cached_engines=reuse_cached_engines,
            engine_cache_size=1 << 30,  # 1GB
        )
        # output = trt_gm(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("----------------dynamo_compile----------------")
    print("disable engine caching, used:", times[0], "ms")
    print("enable engine caching to cache engines, used:", times[1], "ms")
    print("enable engine caching to reuse engines, used:", times[2], "ms")


# Custom Engine Cache
class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_dir: str,
    ) -> None:
        self.engine_cache_dir = engine_cache_dir

    def save(
        self,
        hash: str,
        blob: bytes,
        prefix: str = "blob",
    ):
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

        path = os.path.join(
            self.engine_cache_dir,
            f"{prefix}_{hash}.bin",
        )
        with open(path, "wb") as f:
            f.write(blob)

    def load(self, hash: str, prefix: str = "blob") -> Optional[bytes]:
        path = os.path.join(self.engine_cache_dir, f"{prefix}_{hash}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                blob = f.read()
            return blob
        return None


def torch_compile(iterations=3):
    times = []
    engine_cache = MyEngineCache("/tmp/your_dir")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        # remove timing cache and reset dynamo just for engine caching messurement
        remove_timing_cache()
        torch._dynamo.reset()

        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        compiled_model = torch.compile(
            model,
            backend="tensorrt",
            options={
                "use_python_runtime": True,
                "enabled_precisions": enabled_precisions,
                "debug": debug,
                "min_block_size": min_block_size,
                "make_refitable": True,
                "cache_built_engines": cache_built_engines,
                "reuse_cached_engines": reuse_cached_engines,
                "custom_engine_cache": engine_cache,  # use custom engine cache
            },
        )
        compiled_model(*inputs)  # trigger the compilation
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("----------------torch_compile----------------")
    print("disable engine caching, used:", times[0], "ms")
    print("enable engine caching to cache engines, used:", times[1], "ms")
    print("enable engine caching to reuse engines, used:", times[2], "ms")


if __name__ == "__main__":
    dynamo_compile()
    torch_compile()
