# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import argparse
import logging
from typing import Any, Dict, Optional

import torch
import torch_tensorrt
from diffusers import FluxPipeline, FluxTransformer2DModel
from torch.export import Dim
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import export_llm, generate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

import time
from contextlib import contextmanager


@contextmanager
def timer(logger, name: str):
    logger.info(f"{name} section Start...")
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} section End...")
    logger.info(f"{name} section elapsed time: {end - start} seconds")


class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        **kwargs,
    ):

        return self.module.forward(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
        )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    # The following options are manual user provided settings
    arg_parser.add_argument(
        "--use_fp32_acc",
        action="store_true",
        help="Use FP32 acc",
    )
    arg_parser.add_argument(
        "--save_engine",
        action="store_true",
        help="Just save the TRT engine and stop the program",
    )
    arg_parser.add_argument(
        "--export",
        action="store_true",
        help="Re-export the TRT module",
    )
    args = arg_parser.parse_args()

    # parameter setting
    batch_size = 2
    max_seq_len = 256
    prompt = ["A cat holding a sign that says hello world" for _ in range(batch_size)]
    cuda_device = "cuda:0"
    device = cuda_device

    with torch.no_grad():
        # Define the model
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16
        )
        pipe.to(device)

        example_inputs = (
            torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(device),
            torch.randn((batch_size, 256, 4096), dtype=torch.float16).to(device),
            torch.randn((batch_size, 768), dtype=torch.float16).to(device),
            torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
            torch.randn((batch_size, 4096, 3), dtype=torch.float16).to(device),
            torch.randn((batch_size, 256, 3), dtype=torch.float16).to(device),
        )
        BATCH = Dim("batch", min=1, max=batch_size)
        SEQ_LEN = Dim("seq_len", min=1, max=max_seq_len)
        dynamic_shapes = (
            {0: BATCH},
            {0: BATCH, 1: SEQ_LEN},
            {0: BATCH},
            {0: BATCH},
            {0: BATCH},
            {0: BATCH, 1: SEQ_LEN},
        )
        free, total = torch.cuda.mem_get_info(cuda_device)
        print(f"== After model declaration == Free mem: {free}, Total mem: {total}")

        # Export the transformer
        with timer(logger=logger, name="ep_gen"):
            model = MyModule(pipe.transformer).eval().half().to(device)
            logger.info("Directly use _export because torch.export.export doesn't work")
            # This API is used to express the constraint violation guards as asserts in the graph.
            from torch.export._trace import _export

            ep = _export(
                model,
                args=example_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                allow_complex_guards_as_runtime_asserts=True,
            )
        free, total = torch.cuda.mem_get_info(cuda_device)
        print(f"== After model export == Free mem: {free}, Total mem: {total}")

        # Torch-TensorRT compilation
        logger.info(f"Generating TRT engine now.")
        use_explicit_typing, use_fp32_acc = False, False
        enabled_precisions = {torch.float16}
        if args.use_fp32_acc:
            use_explicit_typing = True
            use_fp32_acc = True
            enabled_precisions = {torch.float32}

        if args.save_engine:
            with torch_tensorrt.logging.debug():
                serialized_engine = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                    ep,
                    inputs=list(example_inputs),
                    enabled_precisions=enabled_precisions,
                    truncate_double=True,
                    device=torch.device(cuda_device),
                    disable_tf32=True,
                    use_explicit_typing=use_explicit_typing,
                    debug=True,
                    use_fp32_acc=use_fp32_acc,
                )
                with open("flux_trt.engine", "wb") as file:
                    file.write(serialized_engine)

                free, total = torch.cuda.mem_get_info(cuda_device)
                print(
                    f"== After saving TRT engine == Free mem: {free}, Total mem: {total}"
                )
        else:
            with timer(logger, "trt_gen"):
                with torch_tensorrt.logging.debug():
                    trt_start = time.time()
                    trt_model = torch_tensorrt.dynamo.compile(
                        ep,
                        inputs=list(example_inputs),
                        enabled_precisions=enabled_precisions,
                        truncate_double=True,
                        device=torch.device(cuda_device),
                        disable_tf32=True,
                        use_explicit_typing=use_explicit_typing,
                        debug=True,
                        use_fp32_acc=use_fp32_acc,
                    )
                    trt_end = time.time()
                    config = pipe.transformer.config
                    pipe.transformer = trt_model
                    pipe.transformer.config = config

                    free, total = torch.cuda.mem_get_info(cuda_device)
                    print(
                        f"== After compiling TRT model and before image gen == Free mem: {free}, Total mem: {total}"
                    )

                    del ep
                    del model
                    print("=== FINISHED TRT COMPILATION. GENERATING IMAGE NOW ...")
                    prompt = "A cat holding a sign that says hello world"
                    image = pipe(
                        prompt,
                        guidance_scale=0.0,
                        num_inference_steps=4,
                        max_sequence_length=128,
                        generator=torch.Generator("cpu").manual_seed(0),
                    ).images[0]
                    image.save("./flux-schnell.png")

                    free, total = torch.cuda.mem_get_info(cuda_device)
                    print(f"== After image gen == Free mem: {free}, Total mem: {total}")

            if args.export:
                with timer(logger, "trt_save"):
                    try:
                        trt_ep = torch.export.export(
                            trt_model,
                            args=example_inputs,
                            dynamic_shapes=dynamic_shapes,
                            strict=False,
                        )
                        torch.export.save(trt_ep, "trt.ep")
                        free, total = torch.cuda.mem_get_info(cuda_device)
                        print(
                            f"== After TRT model re-export == Free mem: {free}, Total mem: {total}"
                        )
                    except Exception as e:
                        import traceback

                        # Capture the full traceback
                        tb = traceback.format_exc()
                        logger.warning("An error occurred. Here's the traceback:")
                        # print(tb)
                        logger.warning(tb)
                        torch_tensorrt.save(trt_model, "trt.ep")
                        free, total = torch.cuda.mem_get_info(cuda_device)
                        print(
                            f"== After saving TRT module via torch_tensorrt.save == Free mem: {free}, Total mem: {total}"
                        )
