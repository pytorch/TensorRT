# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
from time import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../examples/apps"))


@torch.no_grad()
def smoke_test(args):
    """Check a reduced, randomly initialized Flux transformer without a pipeline."""
    if args.dtype != "fp16":
        raise ValueError("--smoke_test currently supports only --dtype fp16")
    if (
        args.num_layers < 0
        or args.num_single_layers < 0
        or args.num_layers + args.num_single_layers == 0
    ):
        raise ValueError("Specify at least one block, with nonnegative layer counts")
    if args.dynamic_shapes or args.use_sdpa:
        raise ValueError("--smoke_test uses static shapes and default Flux attention")

    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    torch.manual_seed(42)
    # Construct only the requested blocks on CPU. Never load the full checkpoint.
    model = (
        FluxTransformer2DModel(
            num_layers=args.num_layers,
            num_single_layers=args.num_single_layers,
            guidance_embeds=True,
        )
        .eval()
        .to(device="cuda:0", dtype=torch.float16)
    )
    config = model.config

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        ):
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

    model = TransformerWrapper(model).eval()
    device = "cuda:0"
    inputs = {
        "hidden_states": torch.randn(
            1, 64, config.in_channels, device=device, dtype=torch.float16
        ),
        "encoder_hidden_states": torch.randn(
            1, 16, config.joint_attention_dim, device=device, dtype=torch.float16
        ),
        "pooled_projections": torch.randn(
            1, config.pooled_projection_dim, device=device, dtype=torch.float16
        ),
        "timestep": torch.ones(1, device=device, dtype=torch.float16),
        "img_ids": torch.zeros(64, 3, device=device, dtype=torch.float16),
        "txt_ids": torch.zeros(16, 3, device=device, dtype=torch.float16),
        "guidance": torch.ones(1, device=device, dtype=torch.float32),
    }
    print(
        f"Random-weight Flux smoke test: {args.num_layers} dual-stream blocks, "
        f"{args.num_single_layers} single-stream blocks"
    )
    expected = model(**inputs)
    if not torch.isfinite(expected).all():
        raise RuntimeError("PyTorch output contains non-finite values")
    if args.pytorch:
        print(f"PyTorch smoke test passed: output shape {tuple(expected.shape)}")
        return

    import torch_tensorrt

    example_inputs = tuple(inputs.values())
    ep = torch.export.export(model, args=example_inputs, strict=False)
    compiled = torch_tensorrt.dynamo.compile(
        ep,
        inputs=example_inputs,
        min_block_size=1,
        require_full_compilation=True,
        use_fp32_acc=True,
        truncate_double=True,
        offload_module_to_cpu=args.low_vram_mode,
        workspace_size=1 << 30,
    )
    actual = compiled(*example_inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(f"TensorRT smoke test passed: output shape {tuple(actual.shape)}")


def benchmark(pipe, prompt, inference_step, batch_size=1, iterations=1):
    print(f"Running warmup with {batch_size=} {inference_step=} iterations=10")
    # warmup
    for i in range(10):
        start = time()
        images = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
        print(
            f"Warmup {i} done in {time() - start} seconds, with {batch_size=} {inference_step=}, generated {len(images)} images"
        )

    # actual benchmark
    print(f"Running benchmark with {batch_size=} {inference_step=} {iterations=}")
    start = time()
    for i in range(iterations):
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
    end = time()
    print(f"Batch Size: {batch_size}")
    print("Time Elapse for", iterations, "iterations:", end - start)
    print(
        "Average Latency Per Step:",
        (end - start) / inference_step / iterations / batch_size,
    )
    return


def main(args):
    print(f"Running flux_perfwith args: {args}")
    if args.smoke_test:
        smoke_test(args)
        return
    if not args.pytorch:
        from flux_demo import compile_model

        pipe, backbone, trt_gm = compile_model(args)
    else:
        from diffusers import FluxPipeline

        pipe = (
            FluxPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
            )
            .to(torch.float16)
            .to("cuda:0")
        )

    benchmark(pipe, ["Test"], 20, batch_size=args.max_batch_size, iterations=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Flux quantization with different dtypes"
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-dev",
        help="Model to use",
    )
    parser.add_argument(
        "--use_sdpa",
        action="store_true",
        help="Use sdpa",
        default=False,
    )
    parser.add_argument(
        "--dtype",
        choices=["fp4", "fp8", "int8", "fp16"],
        default="fp16",
        help="Select the data type to use (fp4 or fp8 or int8 or fp16)",
    )
    parser.add_argument(
        "--fp4_mha",
        action="store_true",
        help="Use NVFP4_FP8_MHA_CONFIG config instead of NVFP4_DEFAULT_CFG",
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        help="Use low VRAM mode when you have a small GPU (<=32GB)",
    )
    parser.add_argument(
        "--dynamic_shapes",
        "-d",
        action="store_true",
        help="Use dynamic shapes",
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Use pytorch runtime and no tensorrt",
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Test a small random-weight transformer without loading the Flux pipeline",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of dual-stream blocks in --smoke_test (default: 1)",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=0,
        help="Number of single-stream blocks in --smoke_test (default: 0)",
    )
    args = parser.parse_args()
    main(args)
