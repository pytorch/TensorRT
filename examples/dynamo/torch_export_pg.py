import torch
import torch_tensorrt
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image

# 1. Model
DEVICE = torch.device("cuda:0")
model_id = "google/paligemma2-3b-pt-224"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)

model = (
    PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    .eval()
    .to(DEVICE)
)
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = ""
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
input_len = model_inputs["input_ids"].shape[-1]

# 2. PyTorch
with torch.inference_mode():
    pyt_generation = model.generate(
        **model_inputs, max_new_tokens=100, do_sample=False
    )  # , use_cache=False)
    # The newly generated tokens after the input tokens.
    pyt_generation = pyt_generation[0][input_len:]
    pyt_decoded = processor.decode(pyt_generation, skip_special_tokens=True)
    print("=============================")
    print("PyTorch generated text:")
    print(pyt_decoded)
    print("=============================")

# (a) Dummy inputs
batch_size = 1
dummy_input_ids = model_inputs["input_ids"]
dummy_attention_mask = model_inputs["attention_mask"]
dummy_pixel_values = model_inputs["pixel_values"]

dummy_inputs = {
    "input_ids": dummy_input_ids,
    "attention_mask": dummy_attention_mask,
    "pixel_values": dummy_pixel_values,
}

# (b) Dynamic shape
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=1024)
dynamic_shapes = {
    "input_ids": {0: BATCH, 1: SEQ_LEN},
    "attention_mask": {0: BATCH, 1: SEQ_LEN},
    "pixel_values": {0: BATCH},
}
# (c) ExportedProgram
# torch.export.export(
#     model,
#     args=(),
#     kwargs=dummy_inputs,
#     dynamic_shapes=dynamic_shapes,
#     strict=False,
# )


import torch
import torch.utils._pytree as pytree
import transformers


def flatten_hybridcache(hc: transformers.cache_utils.HybridCache):
    """
    1) Collects all tensors inside HybridCache into a list.
    2) Stores non-tensor values in the context (dictionary).
    """
    # 1. Tensors: is_sliding, entire key_cache, entire value_cache
    flat_tensors = []
    flat_tensors.append(hc.is_sliding)  # shape: [num_hidden_layers], bool
    flat_tensors.extend(hc.key_cache)  # List[Tensor]
    flat_tensors.extend(hc.value_cache)  # List[Tensor]

    # 2. Store non-tensor fields in the context
    context = {
        "max_cache_len": hc.max_cache_len,
        "max_batch_size": hc.max_batch_size,
        "head_dim": hc.head_dim,
        "dtype": hc.dtype,
        "num_key_value_heads": hc.num_key_value_heads,
        "num_layers": len(
            hc.key_cache
        ),  # = len(hc.value_cache) = config.num_hidden_layers
    }

    return flat_tensors, context


def unflatten_hybridcache(flat_tensors, context):
    """
    Restores a HybridCache object from the (flat_tensors, context) produced by flatten_hybridcache.
    """
    num_layers = context["num_layers"]

    # 1. Parse flat_tensors
    #    - First element is is_sliding
    #    - Next num_layers elements: key_cache
    #    - Next num_layers elements: value_cache
    is_sliding = flat_tensors[0]
    key_cache = flat_tensors[1 : 1 + num_layers]
    value_cache = flat_tensors[1 + num_layers : 1 + 2 * num_layers]

    # 2. Create an empty HybridCache object using __new__ (without calling __init__)
    hc = transformers.cache_utils.HybridCache.__new__(
        transformers.cache_utils.HybridCache
    )

    # 3. Manually set required fields
    hc.max_cache_len = context["max_cache_len"]
    hc.max_batch_size = context["max_batch_size"]
    hc.head_dim = context["head_dim"]
    hc.dtype = context["dtype"]
    hc.num_key_value_heads = context["num_key_value_heads"]
    hc.is_sliding = is_sliding
    hc.key_cache = list(key_cache)
    hc.value_cache = list(value_cache)

    return hc


# Register with pytree
pytree.register_pytree_node(
    transformers.cache_utils.HybridCache, flatten_hybridcache, unflatten_hybridcache
)

# from torch.export._trace import _export
# exported_program = _export(
#     model,
#     args=(),
#     kwargs=dummy_inputs,
#     dynamic_shapes=dynamic_shapes,
#     strict=False,
#     allow_complex_guards_as_runtime_asserts=True,
# )

# torch.export._draft_export.draft_export
import torch.export._draft_export

exported_program = torch.export._draft_export.draft_export(
    model,
    args=(),
    kwargs=dummy_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
    # allow_complex_guards_as_runtime_asserts=True,
)


trt_model = torch_tensorrt.dynamo.compile(
    exported_program[0],
    inputs=dummy_inputs,
    enabled_precisions={torch.float32},
    truncate_double=True,
    device=DEVICE,
    disable_tf32=True,
    use_explicit_typing=True,
    use_fp32_acc=True,
)

# Execute generation using TensorRT model
model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
with torch.inference_mode():
    trt_generation = trt_model.generate(
        **model_inputs, max_new_tokens=100, do_sample=False
    )
    trt_generation = trt_generation[0][input_len:]
    trt_decoded = processor.decode(trt_generation, skip_special_tokens=True)
    print("TensorRT generated text:")
    print(trt_decoded)
