import torch
import copy
import torchvision
import torch_tensorrt
from torch_tensorrt.fx import InputTensorSpec


def test_torch_tensorrt(model, inputs):
    # torchscript path
    model_ts = copy.deepcopy(model)
    inputs_ts = copy.deepcopy(inputs)
    # fp32 test
    with torch.inference_mode():
        ref_fp32 = model_ts(*inputs_ts)
    trt_ts_module = torch_tensorrt.compile(
        model_ts, inputs=inputs_ts, enabled_precisions={torch.float32}
    )
    result_fp32 = trt_ts_module(*inputs_ts)
    assert (
        torch.nn.functional.cosine_similarity(
            ref_fp32.flatten(), result_fp32.flatten(), dim=0
        )
        > 0.9999
    )
    # fp16 test
    model_ts = model_ts.half()
    inputs_ts = [i.cuda().half() for i in inputs_ts]
    with torch.inference_mode():
        ref_fp16 = model_ts(*inputs_ts)
    trt_ts_module = torch_tensorrt.compile(
        model_ts, inputs=inputs_ts, enabled_precisions={torch.float16}
    )
    result_fp16 = trt_ts_module(*inputs_ts)
    assert (
        torch.nn.functional.cosine_similarity(
            ref_fp16.flatten(), result_fp16.flatten(), dim=0
        )
        > 0.99
    )

    # FX path
    model_fx = copy.deepcopy(model)
    inputs_fx = copy.deepcopy(inputs)
    # fp32 test
    with torch.inference_mode():
        ref_fp32 = model_fx(*inputs_fx)
    trt_fx_module = torch_tensorrt.compile(
        model_fx, ir="fx", inputs=inputs_fx, enabled_precisions={torch.float32}
    )
    result_fp32 = trt_fx_module(*inputs_fx)
    assert (
        torch.nn.functional.cosine_similarity(
            ref_fp32.flatten(), result_fp32.flatten(), dim=0
        )
        > 0.9999
    )
    # fp16 test
    model_fx = model_fx.cuda().half()
    inputs_fx = [i.cuda().half() for i in inputs_fx]
    with torch.inference_mode():
        ref_fp16 = model_fx(*inputs_fx)
    trt_fx_module = torch_tensorrt.compile(
        model_fx, ir="fx", inputs=inputs_fx, enabled_precisions={torch.float16}
    )
    result_fp16 = trt_fx_module(*inputs_fx)
    assert (
        torch.nn.functional.cosine_similarity(
            ref_fp16.flatten(), result_fp16.flatten(), dim=0
        )
        > 0.99
    )


if __name__ == "__main__":
    model = torchvision.models.resnet18(pretrained=True).cuda().eval()
    inputs = [torch.ones((32, 3, 224, 224), device=torch.device("cuda"))]  # type: ignore[attr-defined]
    test_torch_tensorrt(model, inputs)
