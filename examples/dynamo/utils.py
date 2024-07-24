import torch
from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model, (inputs,), dynamic_shapes=({1: seq_len},), strict=False
            )
        except:
            print(
                "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
            )
            # This API is used to express the constraint violation guards as asserts in the graph.
            ep = torch.export._trace._export(
                model,
                (inputs,),
                dynamic_shapes=({1: seq_len},),
                strict=False,
                allow_complex_guards_as_runtime_asserts=True,
            )

    return ep


def generate(model, input_seq, max_tokens, eos_token_id):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    max_length = len(input_seq) + max_tokens
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    token_id = 0
    while token_id < max_tokens:
        # print("Generating token: ", token_id)
        outputs = model(input_seq)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        if stopping_criteria(input_seq, logits).item():
            break
        token_id += 1

    return input_seq, token_id
