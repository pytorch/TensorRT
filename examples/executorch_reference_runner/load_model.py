"""Load and optionally run a Torch-TensorRT ExecuTorch ``.pte`` model.

The default input matches ``export_static_shape.py``: one CUDA float tensor
with shape ``(2, 3, 4, 4)`` filled with ones.
"""

import argparse
from pathlib import Path

import torch
from torch_tensorrt.executorch.runtime import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=Path("model.pte"))
    parser.add_argument("--method", default="forward")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--load_only",
        action="store_true",
        help="Parse the program and print its methods without loading/executing one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"ExecuTorch model not found: {args.model_path}")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be at least 1")

    program = load(args.model_path)
    print(f"Loaded {args.model_path}")
    print(f"Program methods: {sorted(program.method_names)}")

    if args.load_only:
        return
    if args.method not in program.method_names:
        raise ValueError(
            f"Method {args.method!r} is not in the program; "
            f"available methods: {sorted(program.method_names)}"
        )
    inputs = (torch.ones((2, 3, 4, 4), dtype=torch.float32),)
    for run in range(args.num_runs):
        outputs = program.run(inputs, args.method)
        print(f"Run {run + 1} outputs:")
        for index, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(
                    f"  output[{index}]: shape={tuple(output.shape)}, "
                    f"dtype={output.dtype}, device={output.device}, "
                    f"sample={output.flatten()[:8]}"
                )
            else:
                print(f"  output[{index}]: {output!r}")


if __name__ == "__main__":
    main()
