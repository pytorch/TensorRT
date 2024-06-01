import pytest
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


@pytest.mark.unit
def test_dynamic_generation_python_rt():
    """
    Tests HuggingFace Generate Code with dynamic shapes
    Code Credit: @peri044
    """
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "gpt2", pad_token_id=tokenizer.eos_token_id, use_cache=False
        )
        .eval()
        .to("cuda")
    )

    # Input prompt
    model_inputs = tokenizer(("Repeat " * 128)[:-1], return_tensors="pt").to("cuda")
    input_ids = model_inputs["input_ids"]
    max_tokens = 40

    # Pyt model outputs
    greedy_output = model.generate(**model_inputs, max_new_tokens=max_tokens)
    print(
        "Pytorch model generated text: ",
        tokenizer.decode(greedy_output[0], skip_special_tokens=True),
    )

    # Compile Torch-TRT model
    torch._dynamo.mark_dynamic(input_ids, 1, min=2, max=1023)
    model.forward = torch.compile(
        model.forward,
        backend="tensorrt",
        dynamic=None,
        options={
            "enabled_precisions": {torch.float},
            "torch_executed_ops": {"torch.ops.aten.slice.Tensor"},
            "use_python_runtime": True,
            "optimization_level": 0,
            "min_block_size": 29,
        },
    )

    # Auto-regressive generation loop for greedy search
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_tokens),
            EosTokenCriteria(eos_token_id=tokenizer.eos_token_id),
        ]
    )
    while True:
        trt_outputs = model(input_ids)
        logits = trt_outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if stopping_criteria(input_ids, logits).item():
            break

    # TODO: Add test for correctness


@pytest.mark.unit
def test_dynamic_generation_cpp_rt():
    """
    Tests HuggingFace Generate Code with dynamic shapes
    Code Credit: @peri044
    """
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "gpt2", pad_token_id=tokenizer.eos_token_id, use_cache=False
        )
        .eval()
        .to("cuda")
    )

    # Input prompt
    model_inputs = tokenizer(("Repeat " * 128)[:-1], return_tensors="pt").to("cuda")
    input_ids = model_inputs["input_ids"]
    max_tokens = 40

    # Pyt model outputs
    greedy_output = model.generate(**model_inputs, max_new_tokens=max_tokens)
    print(
        "Pytorch model generated text: ",
        tokenizer.decode(greedy_output[0], skip_special_tokens=True),
    )

    # Compile Torch-TRT model
    torch._dynamo.mark_dynamic(input_ids, 1, min=2, max=1023)
    model.forward = torch.compile(
        model.forward,
        backend="tensorrt",
        dynamic=None,
        options={
            "enabled_precisions": {torch.float},
            "torch_executed_ops": {"torch.ops.aten.slice.Tensor"},
            "use_python_runtime": False,
            "optimization_level": 0,
            "min_block_size": 29,
        },
    )

    # Auto-regressive generation loop for greedy search
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_tokens),
            EosTokenCriteria(eos_token_id=tokenizer.eos_token_id),
        ]
    )
    while True:
        trt_outputs = model(input_ids)
        logits = trt_outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if stopping_criteria(input_ids, logits).item():
            break

    # TODO: Add test for correctness
