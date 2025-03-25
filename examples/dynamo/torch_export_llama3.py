"""
.. _torch_export_gpt2:

Compiling GPT2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular GPT2 model.
"""

import argparse
import copy
import os
import timeit

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)
from transformers.modeling_utils import load_sharded_checkpoint
from utils import export_llm, generate, recordStats, time_generate

MAX_TOKENS = 100
DEVICE = torch.device("cuda:0")


def _to_maybe_empty(model: torch.nn.Module, device):
    """A mix of ``model.to(device)`` and ``model.to_empty(device)``.

    If a parameter is already initialized, then we will call `to()` on it. Otherwise, we will
    initialize it with an empty tensor on the given device.

    """
    model._apply(
        lambda t: (
            torch.empty_like(t, device=device)
            if t.device == torch.device("meta")
            else t.to(device)
        )
    )


def get_autodeploy_llama3():

    from model import LLamaTransformer, ModelArgs

    default_dtype = torch.get_default_dtype()
    model_kwargs = {
        # "max_position_embeddings": 2176,
        "use_cache": False,
        # 'num_hidden_layers': 2
    }
    model_config = ModelArgs.from_pretrained("llama3-8B", **model_kwargs)
    get_model_from_config = LLamaTransformer.from_config
    torch.set_default_dtype(model_config.torch_dtype)
    model = get_model_from_config(model_config, trust_remote_code=True)
    torch.set_default_dtype(default_dtype)

    if hasattr(model, "post_init"):
        model.post_init()

    model.eval().cuda()
    ckpt_path = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
    if os.path.isfile(os.path.join(ckpt_path, "model.safetensors.index.json")):
        _to_maybe_empty(model, device="cpu")
        load_sharded_checkpoint(model, ckpt_path, strict=False)

    return model.cuda()


def get_model(args):
    with torch.no_grad():
        if args.model == "autodeploy_llama3":
            model = get_autodeploy_llama3()
            model = model.to(torch.float16)
        elif args.model == "meta-llama/Llama-3.2-1B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    num_hidden_layers=2,
                )
                .eval()
                .half()
                .cuda()
            )
            model = model.to(torch.float16)
        elif args.model == "meta-llama/Llama-3.2-3B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    # num_hidden_layers=2
                )
                .eval()
                .half()
                .cuda()
            )
            model = model.to(torch.float16)
        elif args.model == "meta-llama/Llama-3.1-8B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",  # num_hidden_layers=1
                )
                .eval()
                .half()
                .cuda()
            )
            model = model.to(torch.float16)
        elif args.model == "google/gemma-3-1b-it":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    "google/gemma-3-1b-it", use_cache=False, attn_implementation="sdpa"
                )
                .eval()
                .half()
                .cuda()
            )

    return model


def compile_torchtrt(model, input_ids):
    from torch_tensorrt.dynamo.lowering import AttentionRegistry, SequenceInfo

    # Initialize kvcaching
    # construct a cache info object
    MAX_SEQ_LEN = input_ids.shape[-1] + MAX_TOKENS
    sequence_info = SequenceInfo(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        page_size=MAX_SEQ_LEN,
    )
    csi = torch_tensorrt.dynamo.lowering.CachedSequenceInterface(
        AttentionRegistry.get("FlashInfer"), sequence_info=sequence_info, device=DEVICE
    )
    gpt2_ep = export_llm(model, input_ids, max_seq_len=256)
    del model
    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.dynamo.compile(
            gpt2_ep,
            inputs=[input_ids],
            enabled_precisions={torch.float16},
            truncate_double=True,
            device=DEVICE,
            disable_tf32=True,
            # use_explicit_typing=True,
            use_python_runtime=True,
            # use_fp32_acc=True,
            debug=True,
            insert_flashinfer_ops=True,
            cached_seq_interface=csi,
            # min_block_size=1e7,
        )

    return trt_model, csi


def custom_generate(model, input_seq, max_tokens, eos_token_id, csi):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    # Max length of output seq = current input_seq length + max_tokens allowed to generate
    max_output_seq_length = input_seq.shape[1] + max_tokens
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    csi.info.reset()
    csi.info.nest_sequences(input_seq)
    i = 0
    iter_time = []
    while i < MAX_TOKENS:
        sequence_info = csi.info
        start_time = timeit.default_timer()
        # print(f"=== Input shape: {csi.args[0].shape}")
        # breakpoint()
        outputs = model(*csi.args)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        iter_time.append(end_time - start_time)
        logits = csi.info.unnest_sequences(outputs[0])
        logits_last = (
            torch.stack([l_one_seq[-1] for l_one_seq in logits]).unsqueeze(1).float()
        )
        idx_next = logits_last.argmax(dim=-1, keepdim=False)
        sequence_info.update_pos(sequence_info.sequence_lengths)
        sequence_info.nest_sequences(idx_next)
        input_seq = torch.cat([input_seq, idx_next], dim=-1)
        # # TODO: Handle batch in this check
        # if stopping_criteria(input_seq, logits).item():
        #     break
        i += 1

    return input_seq, iter_time


def print_outputs(pyt_gen_tokens, trt_gen_tokens, tokenizer):
    print(pyt_gen_tokens)
    print(trt_gen_tokens)

    print("=============================")
    print(
        "Pytorch model generated text: ",
        tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
    )
    print("=============================")
    print(
        "TensorRT model generated text: ",
        tokenizer.decode(trt_gen_tokens[0], skip_special_tokens=True),
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--model", type=str, default="autodeploy_llama3", help="Name of LLM model"
    )
    arg_parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt", type=str, default="What is parallel programming ?", help="Prompt"
    )
    arg_parser.add_argument("--precision", type=str, default="FP16", help="Prompt")
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    args = arg_parser.parse_args()
    model = get_model(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    prompt = "What is parallel programming ?"
    model_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(DEVICE)

    # Pyt
    pyt_gen_tokens, pyt_iter_time = generate(
        model, input_ids.clone(), MAX_TOKENS, tokenizer.eos_token_id
    )
    pyt_timings = time_generate(
        generate,
        model,
        input_ids.clone(),
        MAX_TOKENS,
        tokenizer.eos_token_id,
        csi=None,
        iterations=args.iterations,
    )
    pyt_stats = recordStats(
        "PyTorch", pyt_timings, args.precision, batch_size=1, compile_time_s=None
    )

    # TRT
    trt_model, csi = compile_torchtrt(model, input_ids)
    trt_gen_tokens, trt_logits = custom_generate(
        trt_model, input_ids.clone(), MAX_TOKENS, tokenizer.eos_token_id, csi
    )

    trt_timings = time_generate(
        custom_generate,
        trt_model,
        input_ids.clone(),
        MAX_TOKENS,
        tokenizer.eos_token_id,
        csi=csi,
        iterations=args.iterations,
    )
    trt_stats = recordStats(
        "TensorRT", trt_timings, args.precision, batch_size=1, compile_time_s=None
    )

    print_outputs(pyt_gen_tokens, trt_gen_tokens, tokenizer)
    print("===================== \n")
    print("=========PyTorch PERFORMANCE============ \n")
    print(pyt_stats)
    print("===================== \n")
    print("=========TensorRT PERFORMANCE============ \n")
    print(trt_stats)
