import os
import sys

import pytest
import torch
import torch_tensorrt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../tools/llm"))
import argparse

from run_llm import compile_torchtrt
from torchtrt_ext import register_sdpa


@pytest.mark.unit
@pytest.mark.parametrize("precision", ["FP16", "BF16", "FP32"])
def test_gemma3_decoder_layer(precision):

    with torch.inference_mode():
        args = argparse.Namespace()
        args.debug = False
        args.num_tokens = 128
        args.model = "google/gemma-3-1b-it"
        args.precision = precision
        args.min_block_size = 1
        args.prompt = "What is parallel programming ?"
        if args.precision == "FP16":
            dtype = torch.float16
        elif args.precision == "BF16":
            dtype = torch.bfloat16
        else:
            args.precision = "FP32"
            dtype = torch.float32

        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
                num_hidden_layers=1,
            )
            .eval()
            .to("cuda")
        )

        register_sdpa._SDPA_MAPPING[args.model](model_config=model.config)
        model = model.to(dtype)
        # use randint will generate nan values in the logits, use a fixed input_ids for now
        # input_ids = torch.randint(0, model.config.vocab_size, (1, args.num_tokens)).to("cuda")
        input_ids = torch.tensor([[2, 3689, 563, 10616, 14929, 2360]]).to("cuda")

        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
        pyt_outputs = model(input_ids.clone(), position_ids=position_ids.clone())
        trt_model = compile_torchtrt(model, input_ids, args)
        trt_outputs = trt_model(input_ids, position_ids=position_ids)

        torch.testing.assert_close(
            pyt_outputs.logits, trt_outputs.logits, rtol=5e-1, atol=5e-1
        )
