import argparse
import os
import sys
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../examples/apps"))
from flux_demo import compile_model


def main(args):
    pipe, backbone, trt_gm = compile_model(args)
    if args.save_full_path:
        trt_ep_path = args.save_full_path
    else:
        trt_ep_path = os.path.join(os.path.dirname(__file__), "flux_trt.ep")
    torch_tensorrt.save(trt_gm, trt_ep_path)
    print(f"Model saved to {trt_ep_path=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Flux quantization with different dtypes"
    )
    parser.add_argument(
        "--fp4_mha",
        action="store_true",
        help="Use FP4 MHA",
    )
    parser.add_argument(
        "--use_explicit_typing",
        action="store_true",
        help="Use explicit typing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp8", "int8", "fp16", "fp4", "bf16"],
        default="fp16",
        help="Select the data type to use (fp8 or int8 or fp16 or fp4 or bf16)",
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
        "--use_torch_dynamo_compile",
        action="store_true",
        help="Use torch.dynamo.compile() to compile the model",
    )
    parser.add_argument(
        "--save_full_path",
        "-s",
        help="Save the model to a full path",
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
