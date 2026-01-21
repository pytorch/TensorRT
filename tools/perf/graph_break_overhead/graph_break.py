import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision
from pyinstrument import Profiler
from torch_tensorrt.dynamo.utils import get_model_device

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import argparse


def benchmark_model(model, input, label, profile=False):
    if profile:
        profiler = Profiler(interval=0.01)
        profiler.start()
    start_time = time.time()
    for _ in range(1000):
        model_outputs = model(*input)
    end_time = time.time()
    print(f"{label} 1000 runs: {end_time - start_time:.4f} seconds")
    if profile:
        profiler.stop()
        profiler.write_html(
            f"/home/other/{label.replace(' ', '_')}.html", timeline=False, show_all=True
        )


def main(args):
    profile = args.profile
    use_python_runtime = args.use_python_runtime
    model_name = args.model

    with torchtrt.dynamo.Debugger(log_level="debug", engine_builder_monitor=False):

        model = (
            torchvision.models.__dict__[model_name](pretrained=True).eval().to("cuda")
        )
        input = [torch.randn((1, 3, 224, 224)).to("cuda")]

        BATCH = torch.export.Dim("BATCH", min=1, max=16)
        exp_program = torch.export.export(model, tuple(input), strict=True)
        trt_mod2 = trt_gm = torchtrt.dynamo.compile(
            exp_program,
            tuple(input),
            use_python_runtime=use_python_runtime,
            enabled_precisions={torch.float},
            min_block_size=1,
            immutable_weights=False,
            reuse_cached_engines=False,
        )

        trt_mod1 = trt_gm = torchtrt.dynamo.compile(
            exp_program,
            tuple(input),
            use_python_runtime=use_python_runtime,
            enabled_precisions={torch.float},
            min_block_size=1,
            immutable_weights=False,
            torch_executed_ops={torch.ops.aten.relu.default},
            reuse_cached_engines=False,
        )

    # AOTI
    if not use_python_runtime:
        torchtrt.save(
            trt_mod1,
            "/home/other/aoti.pt2",
            output_format="aot_inductor",
            inputs=input,
            retrace=True,
        )
        aoti_model_gb = torch._inductor.aoti_load_package("/home/other/aoti.pt2")
        torchtrt.save(
            trt_mod2,
            "/home/other/aoti_no_gb.pt2",
            output_format="aot_inductor",
            inputs=input,
            retrace=True,
        )
        aoti_model_no_gb = torch._inductor.aoti_load_package(
            "/home/other/aoti_no_gb.pt2"
        )

    # Warmup runs to avoid measuring first-run overheads
    for _ in range(100):
        trt_mod2(*input)
        model(*input)
        if not use_python_runtime:
            aoti_model_gb(*input)
            aoti_model_no_gb(*input)

    time.sleep(1)
    benchmark_model(trt_mod1, input, "trt_mod1 (with graph break)", profile=profile)
    benchmark_model(trt_mod2, input, "trt_mod2 (without graph break)", profile=profile)
    if not use_python_runtime:
        benchmark_model(aoti_model_gb, input, "aoti_model_gb", profile=profile)
        benchmark_model(aoti_model_no_gb, input, "aoti_model_no_gb", profile=profile)

    out1 = trt_mod1(*input)
    out2 = trt_mod2(*input)
    if not use_python_runtime:
        out3 = aoti_model_gb(*input)
        out4 = aoti_model_no_gb(*input)

    def _to_tuple(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return (x,)

    outs1 = _to_tuple(out1)
    outs2 = _to_tuple(out2)
    if not use_python_runtime:
        outs3 = _to_tuple(out3)
        outs4 = _to_tuple(out4)

    def compare_outputs(a, b, name1="A", name2="B"):
        if len(a) != len(b):
            print(f"Number of outputs differ: {len(a)} vs {len(b)}")
            return False
        all_equal = True
        for i, (x, y) in enumerate(zip(a, b)):
            if not torch.allclose(x, y, atol=1e-3, rtol=1e-3):
                print(f"Output {i} differs between {name1} and {name2}")
                print(f"max diff: {torch.max(torch.abs(x - y))}")
                print(f"Mean diff: {torch.mean(torch.abs(x - y))}")
                all_equal = False
        if all_equal:
            print(f"All outputs match between {name1} and {name2}")
        return all_equal

    compare_outputs(outs1, outs2, "trt_mod1", "trt_mod2")
    if not use_python_runtime:
        compare_outputs(outs1, outs3, "trt_mod1", "aoti_model_gb")
        compare_outputs(outs1, outs4, "trt_mod1", "aoti_model_no_gb")
        compare_outputs(outs2, outs3, "trt_mod2", "aoti_model")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--profile", action="store_true")
    arg_parser.add_argument("--use_python_runtime", action="store_true")
    arg_parser.add_argument(
        "--model", type=str, default="resnet18", choices=["resnet18", "resnet152"]
    )
    args = arg_parser.parse_args()
    main(args)
