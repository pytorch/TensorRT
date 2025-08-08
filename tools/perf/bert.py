import tokenize
from pickle import FALSE, TRUE
from time import perf_counter

import torch
import torch_tensorrt
from torch.export import Dim
from transformers import BertModel, BertTokenizer

dtypes = [torch.float32, torch.float16, torch.bfloat16]
dynamic_shape_supports = [False]
iterations = 20

results = {}


def run_bert(dtype: torch.dtype):
    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval().to(dtype)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    inputs = (input_ids, token_type_ids)
    exp_program = torch.export.export(model, args=inputs)
    compile_start = perf_counter()
    compiled_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=inputs,
        use_explicit_type=True,
        min_block_size=1,
        immutable_weights=True,
        cache_built_engines=False,
        reuse_cached_engines=False,
        use_python_runtime=False,
    )
    compile_end = perf_counter()
    compile_time = compile_end - compile_start

    with torch.no_grad():
        start = perf_counter()
        compiled_model(*inputs)
        end = perf_counter()
        first_inference_time = end - start
        start = perf_counter()
        for i in range(iterations):
            compiled_model(*inputs)
        end = perf_counter()
        second_inference_time = (end - start) / iterations
    return compile_time, first_inference_time, second_inference_time


for dtype in dtypes:
    results[dtype] = {}
    for dynamic_shape_support in dynamic_shape_supports:
        results[dtype][dynamic_shape_support] = {}
        total_compile_time = 0
        total_first_inference_time = 0
        total_second_inference_time = 0
        for i in range(iterations):
            compile_time, first_inference_time, second_inference_time = run_bert(dtype)
            total_compile_time += compile_time
            total_first_inference_time += first_inference_time
            total_second_inference_time += second_inference_time

        results[dtype][dynamic_shape_support]["compile_time"] = (
            total_compile_time / iterations
        )
        results[dtype][dynamic_shape_support]["first_inference_time"] = (
            total_first_inference_time / iterations
        )
        results[dtype][dynamic_shape_support]["second_inference_time"] = (
            total_second_inference_time / iterations
        )

for dype in dtypes:
    for dynamic_shape_support in dynamic_shape_supports:
        print(f"")
        print(
            f"================================================================================"
        )
        print(f"========Bert model: {dtype=} {dynamic_shape_support=}=========")
        print(f"compile_time: {results[dtype][dynamic_shape_support]['compile_time']}")
        print(
            f"first_inference_time: {results[dtype][dynamic_shape_support]['first_inference_time']}"
        )
        print(
            f"second_inference_time: {results[dtype][dynamic_shape_support]['second_inference_time']}"
        )
        print(
            f"================================================================================"
        )
        print(
            f"================================================================================"
        )
        print(f"")
