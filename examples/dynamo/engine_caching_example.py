import ast
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_caching import BaseEngineCache

_LOGGER: logging.Logger = logging.getLogger(__name__)


np.random.seed(0)
torch.manual_seed(0)
size = (100, 3, 224, 224)

model = models.resnet18(pretrained=True).eval().to("cuda")
enabled_precisions = {torch.float}
debug = False
min_block_size = 1
use_python_runtime = False


def remove_timing_cache(path=TIMING_CACHE_PATH):
    if os.path.exists(path):
        os.remove(path)


def dynamo_path(iterations=3):
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
        remove_timing_cache()  # remove timing cache for engine caching messurement
        if i == 0:
            save_engine_cache = False
            load_engine_cache = False
        else:
            save_engine_cache = True
            load_engine_cache = True

        start.record()
        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(inputs),
            use_python_runtime=use_python_runtime,
            enabled_precisions=enabled_precisions,
            debug=debug,
            min_block_size=min_block_size,
            make_refitable=True,
            save_engine_cache=save_engine_cache,
            load_engine_cache=load_engine_cache,
            engine_cache_size=1 << 30,  # 1GB
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("-----dynamo_path-----> compilation time:\n", times, "milliseconds")


# Custom Engine Cache
class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_size: int,
        engine_cache_dir: str,
    ) -> None:
        self.total_engine_cache_size = engine_cache_size
        self.available_engine_cache_size = engine_cache_size
        self.engine_cache_dir = engine_cache_dir

    def save(
        self,
        hash: str,
        serialized_engine: bytes,
        input_names: List[str],
        output_names: List[str],
    ) -> bool:
        path = os.path.join(
            self.engine_cache_dir,
            f"{hash}/engine--{input_names}--{output_names}.trt",
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(serialized_engine)
        except Exception as e:
            _LOGGER.warning(f"Failed to save the TRT engine to {path}: {e}")
            return False

        _LOGGER.info(f"A TRT engine was cached to {path}")
        serialized_engine_size = int(serialized_engine.nbytes)
        self.available_engine_cache_size -= serialized_engine_size
        return True

    def load(self, hash: str) -> Tuple[Optional[bytes], List[str], List[str]]:
        directory = os.path.join(self.engine_cache_dir, hash)
        if os.path.exists(directory):
            engine_list = os.listdir(directory)
            assert (
                len(engine_list) == 1
            ), f"There are more than one engine {engine_list} under {directory}."
            path = os.path.join(directory, engine_list[0])
            input_names_str, output_names_str = (
                engine_list[0].split(".trt")[0].split("--")[1:]
            )
            input_names = ast.literal_eval(input_names_str)
            output_names = ast.literal_eval(output_names_str)
            with open(path, "rb") as f:
                serialized_engine = f.read()
                return serialized_engine, input_names, output_names
        else:
            return None, [], []


def compile_path(iterations=3):
    times = []
    engine_cache = MyEngineCache(200 * (1 << 20), "/tmp/your_dir")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand(size).to("cuda")]
        # remove timing cache and reset dynamo for engine caching messurement
        remove_timing_cache()
        torch._dynamo.reset()

        if i == 0:
            save_engine_cache = False
            load_engine_cache = False
        else:
            save_engine_cache = True
            load_engine_cache = True

        start.record()
        compiled_model = torch.compile(
            model,
            backend="tensorrt",
            options={
                "use_python_runtime": use_python_runtime,
                "enabled_precisions": enabled_precisions,
                "debug": debug,
                "min_block_size": min_block_size,
                "make_refitable": True,
                "save_engine_cache": save_engine_cache,
                "load_engine_cache": load_engine_cache,
                "engine_cache_instance": engine_cache,  # use custom engine cache
            },
        )
        compiled_model(*inputs)  # trigger the compilation
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("-----compile_path-----> compilation time:\n", times, "milliseconds")


if __name__ == "__main__":
    dynamo_path()
    compile_path()
