import argparse
import copy
import gc
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
from transformers import BertConfig, ReformerConfig, XLNetModel, XLNetConfig

import torchdynamo
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.training import aot_autograd_debug_strategy1
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
from torchdynamo.testing import collect_results
from torchdynamo.testing import same

torch.backends.cuda.matmul.allow_tf32 = True


# This example is for testing the hugging face models. Since the model can not be directly traced by acc tracer(based on torch.fx)
# We combined our efforts together with TorchDynamo. To illustrate the performance, we tested the performance with different batch size.

benchmarks = [
    # Longformer is not suitable for torch_tensorrt-fx
    # (
    #     AutoConfig.from_pretrained("allenai/longformer-base-4096"),
    #     AutoModelForMaskedLM,
    #     (2, 1024),
    #     [torch.bfloat16], # trilu not implemented for bfloat16
    # ),
    # (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []), # Reformer is not suitable for torch_tensorrt-fx
    # (BigBirdConfig(attention_type="block_sparse"), AutoModelForMaskedLM, (2, 1024), []), # Birdbird is not suitable for torch_tensorrt-fx
    # (AutoConfig.from_pretrained("google/fnet-base"), AutoModelForMaskedLM, (4, 512), []), #  not supported by torch_tensorrt-fx
    # batch size = 1
    (BertConfig(), AutoModelForMaskedLM, (1, 512), []),
    (AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (1, 512), []),
    (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (1, 512), []),
    (AutoConfig.from_pretrained("t5-small"), AutoModelForSeq2SeqLM, (1, 512), []),
    (
        AutoConfig.from_pretrained("distilbert-base-uncased"),
        AutoModelForMaskedLM,
        (1, 512),
        [],
    ),
    (AutoConfig.from_pretrained("roberta-base"), AutoModelForMaskedLM, (1, 512), []),
    (AutoConfig.from_pretrained("distilgpt2"), AutoModelForCausalLM, (1, 512), []),
    (
        AutoConfig.from_pretrained("google/electra-base-discriminator"),
        AutoModelForMaskedLM,
        (1, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        (1, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("google/mobilebert-uncased"),
        AutoModelForMaskedLM,
        (1, 512),
        [],
    ),
    (AutoConfig.from_pretrained("camembert-base"), AutoModelForMaskedLM, (1, 512), []),
    (
        AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"),
        AutoModelForMaskedLM,
        (1, 512),
        [],
    ),
    # batch size = 4
    (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    (AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (4, 512), []),
    (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    (AutoConfig.from_pretrained("t5-small"), AutoModelForSeq2SeqLM, (4, 512), []),
    (
        AutoConfig.from_pretrained("distilbert-base-uncased"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    (AutoConfig.from_pretrained("roberta-base"), AutoModelForMaskedLM, (4, 512), []),
    (AutoConfig.from_pretrained("distilgpt2"), AutoModelForCausalLM, (4, 512), []),
    (
        AutoConfig.from_pretrained("google/electra-base-discriminator"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("google/mobilebert-uncased"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    (AutoConfig.from_pretrained("camembert-base"), AutoModelForMaskedLM, (4, 512), []),
    (
        AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"),
        AutoModelForMaskedLM,
        (4, 512),
        [],
    ),
    # batch size = 8
    (BertConfig(), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (8, 512), []),
    (AutoConfig.from_pretrained("t5-small"), AutoModelForSeq2SeqLM, (8, 512), []),
    (
        AutoConfig.from_pretrained("distilbert-base-uncased"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    (AutoConfig.from_pretrained("roberta-base"), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("distilgpt2"), AutoModelForCausalLM, (8, 512), []),
    (
        AutoConfig.from_pretrained("google/electra-base-discriminator"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    (
        AutoConfig.from_pretrained("google/mobilebert-uncased"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
    (AutoConfig.from_pretrained("camembert-base"), AutoModelForMaskedLM, (8, 512), []),
    (
        AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"),
        AutoModelForMaskedLM,
        (8, 512),
        [],
    ),
]

device = "cuda"


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@torchdynamo.skip
def get_cur_memory():
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.current"]
    # print(f"Current memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    return peak_bytes_requirement


@torchdynamo.skip
def forward_pass(mod, inputs, collect_outputs=True):
    return mod(*inputs)


# correctness function to compare with eager mode
@torchdynamo.skip
def check_correctness(args, mod, inputs, optimize_ctx, optimize_name):
    torch.manual_seed(1337)
    correct_result = forward_pass(copy.deepcopy(mod), inputs)

    torch.manual_seed(1337)
    correct_rerun_result = forward_pass(copy.deepcopy(mod), inputs)

    if not same(correct_result, correct_rerun_result):
        print("INCORRECT - Variation in Eager runs itself")
        return False

    torch.manual_seed(1337)
    torchdynamo.reset()
    try:
        with optimize_ctx:
            new_result = forward_pass(mod, inputs)
    except Exception:
        print("ERROR")
        return False

    if optimize_name == "dynamo_fx2trt_fp16":
        cos_similarity = True
    else:
        cos_similarity = False

    if not same(correct_result, new_result, cos_similarity=cos_similarity, tol=1e-2):
        print("INCORRECT")
        return False
    return True


synchronize = torch.cuda.synchronize

# timing function to record the repeated run time
def timed(model, model_iter_fn, train_inputs, timings=1, return_result=False):
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(timings):
        result = model_iter_fn(model, train_inputs, collect_outputs=False)
        synchronize()
    t1 = time.perf_counter()
    # print("===timed=", t1-t0)
    return (t1 - t0, result) if return_result else t1 - t0


# benchmark functions for repeated run of hugging face models after tracing by torchdynamo and lowered through torch_tensorrt-fx
@torchdynamo.skip
def bench_model_eval(args, name, mod, eval_inputs, optimize_ctx):
    if type(optimize_ctx) == NullContext:
        # Profile memory
        m = None
        for i in range(5):
            out = mod(*eval_inputs)
            if i == 4:
                m = get_cur_memory()

        # Warmup
        iters = 5
        for _ in range(iters):
            timed(mod, forward_pass, eval_inputs)
        synchronize()

        # Profile time
        iters = 50
        synchronize()
        timings = []
        for _ in range(iters):
            timings.append(timed(mod, forward_pass, eval_inputs))
        t = np.median(timings, axis=0)
    else:
        # does not need recompile for torchdynamo, demo for fx2trt only
        with torchdynamo.run():
            # Profile memory
            m = None
            for i in range(5):
                out = mod(*eval_inputs)
                if i == 4:
                    m = get_cur_memory()

            # Warmup
            iters = 5
            for _ in range(iters):
                timed(mod, forward_pass, eval_inputs)
            synchronize()

            # Profile time
            iters = 50
            synchronize()
            timings = []
            for _ in range(iters):
                timings.append(timed(mod, forward_pass, eval_inputs))
            t = np.median(timings, axis=0)

    print(name, t, m)
    return t, m


model_header, dtype_header, nh, th, mh, sp, mp, acc = (
    "model",
    "dtype",
    "name",
    "time (s)",
    "mem (GB)",
    "speedup",
    "mem_compression",
    "is_accurate",
)


def create_record(model_name, dtype, is_accurate, name, t, m):
    return {
        model_header: model_name,
        dtype_header: str(dtype),
        acc: is_accurate,
        nh: name,
        th: t,
        mh: m / 2**30,
    }


numerical_diffs = []
results = []


def load_model(config, model_type, dtype, args):
    for attr in dir(config):
        if "drop" in attr and isinstance(getattr(config, attr), float):
            setattr(
                config, attr, 1e-30
            )  # So we can check for correct gradients without eliminating the dropout computation
    model = model_type.from_config(config).to(device, dtype=dtype)
    model.eval()
    return model


class ArgsToKwargsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ArgsToKwargsWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def run_all_eval(args, optimize_ctx, optimize_name, dtype):
    for config, model_type, input_size, not_supported_dtypes in benchmarks:
        if dtype in not_supported_dtypes:
            continue

        model = load_model(config, model_type, dtype, args)

        model_name = type(model).__name__

        # Prepare inputs
        input_ids = torch.randint(0, config.vocab_size, input_size).to(device)

        if model_type.__name__ == "AutoModelForSeq2SeqLM":
            model = ArgsToKwargsWrapper(model)
            eval_inputs = (
                input_ids,
                input_ids,
            )
        else:
            eval_inputs = (input_ids,)

        # Correctness check
        is_accurate = check_correctness(
            args, model, eval_inputs, optimize_ctx, optimize_name
        )
        # Profile eager
        t, m = bench_model_eval(args, "eager", model, eval_inputs, NullContext())
        results.append(create_record(model_name, dtype, is_accurate, "eager", t, m))

        # Profile Dynamo nvfuser
        t, m = bench_model_eval(args, optimize_name, model, eval_inputs, optimize_ctx)
        results.append(
            create_record(model_name, dtype, is_accurate, optimize_name, t, m)
        )

        # calculate relative improvements
        base_r = results[-2]
        for r in results[-2:]:
            r[sp] = round(base_r[th] / r[th], 3)
            r[mp] = round(base_r[mh] / r[mh], 3)
        print(pd.DataFrame(results[-2:]).to_markdown(index=False, floatfmt=".3f"))

    print("=== Final results ===")
    print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-dynamo-eager",
        action="store_true",
        help="Use Dynamo eager",
    )
    group.add_argument(
        "--run-dynamo-fx2trt-fp16",
        action="store_true",
        help="Use Dynamo with fx2trt fp16",
    )
    group.add_argument(
        "--run-dynamo-fx2trt-fp32",
        action="store_true",
        help="Use Dynamo with fx2trt fp32",
    )
    args = parser.parse_args()
    optimize_ctx = NullContext()
    optimize_name = "eager"

    if args.run_dynamo_eager:
        optimize_ctx = torchdynamo.optimize("eager")
        optimize_name = "dynamo_eager"
    elif args.run_dynamo_fx2trt_fp16:
        optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler_fp16)
        optimize_name = "dynamo_fx2trt_fp16"
    elif args.run_dynamo_fx2trt_fp32:
        optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler)
        optimize_name = "dynamo_fx2trt_fp32"

    experiment = run_all_eval
    # fp16
    if optimize_name == "dynamo_fx2trt_fp16":
        experiment = partial(experiment, dtype=torch.float16)
    if optimize_name == "dynamo_fx2trt_fp32":
        experiment = partial(experiment, dtype=torch.float32)

    experiment = partial(
        experiment, optimize_ctx=optimize_ctx, optimize_name=optimize_name
    )
    experiment(args)


if __name__ == "__main__":
    main()
