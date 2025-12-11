# type: ignore
import importlib
import platform
import unittest
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import (
    COSINE_THRESHOLD,
    cosine_similarity,
    get_model_device,
)

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models
if importlib.util.find_spec("timm"):
    import timm


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


def compile_one(idx: int, ir: str):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((idx + 1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"In multiprocess compilation test, process {idx} failed: Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_multiprocess(ir):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    procs = []
    for i in range(3):
        p = mp.Process(target=compile_one, args=(i, ir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_cpu_offload(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "offload_module_to_cpu": True,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    if ir == "dynamo":
        assertions.assertTrue(
            get_model_device(model).type == "cpu",
            msg="Model should be offloaded to CPU",
        )
        model.cuda()
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


# TODO: remove this skip in windows once the access violation issue is fixed
# nvbug: https://nvbugspro.nvidia.com/bug/5555263
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
@unittest.skipIf(
    platform.system().lower().startswith("windows"),
    "Windows cu130 has access violation issue with this test case, skip it for now",
)
def test_resnet18_torch_exec_ops(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(8, 3, 224, 224),
                max_shape=(16, 3, 224, 224),
                dtype=torch.float32,
            )
        ],
        "ir": ir,
        "use_explicit_typing": False,
        "enabled_precisions": {torch.float32, torch.float16},
        "min_block_size": 1,
        "output_format": "exported_program",
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "torch_executed_ops": {torch.ops.aten.matmul, "torch.ops.aten.add"},
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_mobilenet_v2(ir, dtype):
    if torchtrt.ENABLED_FEATURES.tensorrt_rtx and dtype == torch.bfloat16:
        pytest.skip("TensorRT-RTX does not support bfloat16")

    model = models.mobilenet_v2(pretrained=True).eval().to("cuda").to(dtype)
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(dtype)

    compile_spec = {
        "inputs": [
            torchtrt.Input(input.shape, dtype=dtype, format=torch.contiguous_format)
        ],
        "device": torchtrt.Device("cuda:0"),
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    pyt_output = model(input)
    trt_output = trt_mod(input)
    assert pyt_output.dtype == trt_output.dtype
    assert pyt_output.dtype == dtype
    cos_sim = cosine_similarity(pyt_output, trt_output)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@unittest.skipIf(
    not importlib.util.find_spec("timm") or not importlib.util.find_spec("torchvision"),
    "timm or torchvision not installed",
)
def test_efficientnet_b0(ir, dtype):
    if torchtrt.ENABLED_FEATURES.tensorrt_rtx and dtype == torch.bfloat16:
        pytest.skip("TensorRT-RTX does not support bfloat16")

    model = (
        timm.create_model("efficientnet_b0", pretrained=True)
        .eval()
        .to("cuda")
        .to(dtype)
    )
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(dtype)

    compile_spec = {
        "inputs": [
            torchtrt.Input(input.shape, dtype=dtype, format=torch.contiguous_format)
        ],
        "device": torchtrt.Device("cuda:0"),
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    pyt_output = model(input)
    trt_output = trt_mod(input)
    assert pyt_output.dtype == trt_output.dtype
    assert pyt_output.dtype == dtype
    cos_sim = cosine_similarity(pyt_output, trt_output)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
def test_bert_base_uncased(ir, dtype):
    if torchtrt.ENABLED_FEATURES.tensorrt_rtx and dtype == torch.bfloat16:
        pytest.skip("TensorRT-RTX does not support bfloat16")

    from transformers import BertModel

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval().to(dtype)
    input = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
        ],
        "device": torchtrt.Device("cuda:0"),
        "truncate_double": True,
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 15,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
    }
    trt_mod = torchtrt.compile(model, **compile_spec)

    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    for key in model_outputs.keys():
        out, trt_out = model_outputs[key], trt_model_outputs[key]
        assert out.dtype == trt_out.dtype
        assert out.dtype == dtype
        cos_sim = cosine_similarity(out, trt_out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_bert_base_uncased_cpu_offload(ir):
    from transformers import BertModel

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    input = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "truncate_double": True,
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 15,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "offload_module_to_cpu": True,
    }
    trt_mod = torchtrt.compile(model, **compile_spec)
    if ir == "dynamo":
        assertions.assertTrue(
            get_model_device(model).type == "cpu",
            msg="Model should be offloaded to CPU",
        )
        model.cuda()

    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    for key in model_outputs.keys():
        out, trt_out = model_outputs[key], trt_model_outputs[key]
        cos_sim = cosine_similarity(out, trt_out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_half(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda").half()
    input = torch.randn((1, 3, 224, 224)).to("cuda").half()

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.half, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "use_explicit_typing": False,
        "enabled_precisions": {torch.half},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_cosmos_true_div(ir):
    class CosmosLearnablePositionalEmbed(torch.nn.Module):
        def __init__(
            self,
            hidden_size: int,
            max_size: tuple[int, int, int],
            patch_size: tuple[int, int, int],
            eps: float = 1e-6,
        ) -> None:
            super().__init__()

            self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
            self.patch_size = patch_size
            self.eps = eps

            self.pos_emb_t = nn.Parameter(torch.randn(self.max_size[0], hidden_size))
            self.pos_emb_h = nn.Parameter(torch.randn(self.max_size[1], hidden_size))
            self.pos_emb_w = nn.Parameter(torch.randn(self.max_size[2], hidden_size))

        def forward(
            self,
            hidden_states: torch.Tensor,
            num_ranks: Optional[int] = None,
            rank_id: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            pe_size = [
                num_frames // self.patch_size[0],
                height // self.patch_size[1],
                width // self.patch_size[2],
            ]
            if num_ranks is not None and rank_id is not None:
                pe_size[0] = pe_size[0] * num_ranks

            # Use expand() instead of repeat() - torch_tensorrt compatible
            # expand() creates a view without copying data, better for dynamic shapes
            emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].expand(
                batch_size, -1, pe_size[1], pe_size[2], -1
            )
            emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].expand(
                batch_size, pe_size[0], -1, pe_size[2], -1
            )
            emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].expand(
                batch_size, pe_size[0], pe_size[1], -1, -1
            )
            emb = emb_t + emb_h + emb_w
            emb = emb.flatten(1, 3)

            norm = torch.linalg.vector_norm(
                emb, dim=-1, keepdim=True, dtype=torch.float32
            )
            alpha = (norm.numel() / emb.numel()) ** 0.5
            # hidden_size = emb.shape[-1]
            # alpha = (1.0 / hidden_size) ** 0.5
            norm = torch.add(self.eps, norm, alpha=alpha)
            return (emb / norm).type_as(hidden_states)

    with torch.no_grad():
        hidden_states = torch.randn(1, 16, 16, 88, 160).cuda()
        model = CosmosLearnablePositionalEmbed(
            hidden_size=4096,
            max_size=(128, 240, 240),
            patch_size=(1, 2, 2),
        )
        model.eval().cuda()
        pyt_output = model(hidden_states)
        num_latent_frames = torch.export.Dim("num_latent_frames", min=1, max=16)

        ep = torch.export.export(
            model,
            args=(hidden_states,),
            dynamic_shapes=({2: num_latent_frames},),  # Make dimension 2 dynamic
            strict=False,
        )
        trt_model = torchtrt.dynamo.compile(
            ep,
            inputs=(hidden_states,),
            enabled_precisions={torch.bfloat16},
            use_explicit_typing=False,
            use_fp32_acc=False,
            device="cuda:0",
            disable_tf32=True,
            use_python_runtime=True,
            min_block_size=1,
        )
        trt_output = trt_model(hidden_states)

    cos_sim = cosine_similarity(pyt_output, trt_output)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Cosmos Learnable Positional Embed TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    torchtrt.ENABLED_FEATURES.tensorrt_rtx,
    "bf16 is not supported for tensorrt_rtx",
)
@pytest.mark.critical
def test_bf16_model(ir):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            return out

    model = MyModule().eval().cuda().to(torch.bfloat16)
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(torch.bfloat16)

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.bfloat16, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float32},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))

    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"BF16 model TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    torchtrt.ENABLED_FEATURES.tensorrt_rtx,
    "bf16 is not supported for tensorrt_rtx",
)
@pytest.mark.critical
def test_bf16_fallback_model(ir):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, stride=1, bias=True)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True)

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            out = self.conv2(out)
            return out

    model = MyModule().eval().cuda().to(torch.bfloat16)
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(torch.bfloat16)

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.bfloat16, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float32},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
        "torch_executed_ops": {"torch.ops.aten.relu.default"},
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))

    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"BF16 fallback model TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()
