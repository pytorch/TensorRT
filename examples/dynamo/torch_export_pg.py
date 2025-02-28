import torch
import torch_tensorrt
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image


# 1. Model
DEVICE = torch.device("cuda:0")
model_id = "google/paligemma2-3b-pt-224"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16
).eval().to(DEVICE)
processor = PaliGemmaProcessor.from_pretrained(model_id)

prompt = ""
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
input_len = model_inputs["input_ids"].shape[-1]

# 2. PyTorch
with torch.inference_mode():
    pyt_generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False) #, use_cache=False)
    # 입력 토큰 이후의 새로 생성된 토큰만 취합니다.
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
    1) HybridCache 내부의 '텐서'들을 리스트로 모은다.
    2) 텐서가 아닌 값들은 context(dict)에 담는다.
    """
    # 1. 텐서로 취급할 것들: is_sliding, key_cache 전체, value_cache 전체
    flat_tensors = []
    flat_tensors.append(hc.is_sliding)               # shape: [num_hidden_layers], bool
    flat_tensors.extend(hc.key_cache)                # List[Tensor]
    flat_tensors.extend(hc.value_cache)              # List[Tensor]

    # 2. 텐서가 아닌 필드는 context로 저장
    context = {
        "max_cache_len": hc.max_cache_len,
        "max_batch_size": hc.max_batch_size,
        "head_dim": hc.head_dim,
        "dtype": hc.dtype,
        "num_key_value_heads": hc.num_key_value_heads,
        # unflatten 시에 key_cache / value_cache를 몇 개씩 떼어낼지 알아야 하므로
        "num_layers": len(hc.key_cache),  # = len(hc.value_cache) = config.num_hidden_layers
    }

    return flat_tensors, context


def unflatten_hybridcache(flat_tensors, context):
    """
    flatten_hybridcache에서 분리한 (flat_tensors, context)를 받아
    다시 HybridCache 객체로 복원하는 함수.
    """
    num_layers = context["num_layers"]

    # 1. flat_tensors 파싱
    #    - 첫 번째 요소가 is_sliding
    #    - 그 다음 num_layers개: key_cache
    #    - 그 다음 num_layers개: value_cache
    is_sliding = flat_tensors[0]
    key_cache = flat_tensors[1 : 1 + num_layers]
    value_cache = flat_tensors[1 + num_layers : 1 + 2*num_layers]

    # 2. __new__로 빈 HybridCache 객체 생성 (생성자 __init__은 호출 안 함)
    hc = transformers.cache_utils.HybridCache.__new__(transformers.cache_utils.HybridCache)

    # 3. 필요한 필드를 직접 셋팅
    hc.max_cache_len = context["max_cache_len"]
    hc.max_batch_size = context["max_batch_size"]
    hc.head_dim = context["head_dim"]
    hc.dtype = context["dtype"]
    hc.num_key_value_heads = context["num_key_value_heads"]
    hc.is_sliding = is_sliding
    hc.key_cache = list(key_cache)
    hc.value_cache = list(value_cache)

    return hc

# pytree 등록
pytree.register_pytree_node(
    transformers.cache_utils.HybridCache,
    flatten_hybridcache,
    unflatten_hybridcache
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
    use_fp32_acc=True,  # FP32 누적을 사용해 정확도를 보존합니다.
)

# ----------------------------
# 5. TensorRT 모델로 생성 수행
# ----------------------------
# (원래의 모델 입력을 GPU로 이동시킨 후 generate() 호출)
model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
with torch.inference_mode():
    trt_generation = trt_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    trt_generation = trt_generation[0][input_len:]
    trt_decoded = processor.decode(trt_generation, skip_special_tokens=True)
    print("TensorRT generated text:")
    print(trt_decoded)


# pytree._register_pytree_node(transformers.modeling_outputs.MaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.MaskedLMOutput(loss=values[0], logits=values[1]))