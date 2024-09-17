"""

.. _engine_caching_bert_example:

Engine Caching (BERT)
=======================

Small caching example on BERT.
"""

import numpy as np
import torch
import torch_tensorrt
from engine_caching_example import remove_timing_cache
from transformers import BertModel

np.random.seed(0)
torch.manual_seed(0)

model = BertModel.from_pretrained("bert-base-uncased", return_dict=False).cuda().eval()
inputs = [
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
]


def compile_bert(iterations=3):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        # remove timing cache and reset dynamo for engine caching messurement
        remove_timing_cache()
        torch._dynamo.reset()

        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        compilation_kwargs = {
            "use_python_runtime": False,
            "enabled_precisions": {torch.float},
            "truncate_double": True,
            "debug": False,
            "min_block_size": 1,
            "make_refittable": True,
            "cache_built_engines": cache_built_engines,
            "reuse_cached_engines": reuse_cached_engines,
            "engine_cache_dir": "/tmp/torch_trt_bert_engine_cache",
            "engine_cache_size": 1 << 30,  # 1GB
        }
        optimized_model = torch.compile(
            model,
            backend="torch_tensorrt",
            options=compilation_kwargs,
        )
        optimized_model(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("-----compile bert-----> compilation time:\n", times, "milliseconds")


if __name__ == "__main__":
    compile_bert()
