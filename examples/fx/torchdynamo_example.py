import typing as t
from copy import deepcopy
from dataclasses import dataclass, field, replace

import torch
import torchdynamo
import torchvision
from torch_tensorrt.fx.lower import compile
from torch_tensorrt.fx.utils import LowerPrecision
from torchdynamo.optimizations import backends

"""
The purpose of this example is to demostrate the lowering flow to TRT and Torchdynamo
To install Torchdynamo, download and run command `python setup.py develop`(https://github.com/facebookresearch/torchdynamo)
"""


@dataclass
class Configuration:
    """
    Specify the configuration used for fx2trt lowering and benchmark.

    To extend, add a new configuration field to this class, and modify the
    lowering or benchmark behavior in `run_configuration_benchmark()`
    correspondingly.

    It automatically prints all its values thanks to being a dataclass.
    """

    # number of inferences to run
    batch_iter: int

    # Input batch size
    batch_size: int

    # Friendly name of the configuration
    name: str = ""

    # Whether to apply TRT lowering to the model before benchmarking
    trt: bool = False

    # Whether to apply torchdynamo
    torchdynamo: bool = False

    # Whether to enable FP16 mode for TRT lowering
    fp16: bool = False

    # Relative tolerance for accuracy check after lowering. -1 means do not
    # check accuracy.
    accuracy_rtol: float = -1  # disable


@dataclass
class Result:
    """Holds and computes the benchmark results.

    Holds raw essential benchmark result values like duration.
    Also computes results that can be derived from the raw essential values
    (QPS), in the form of auto properties.

    """

    module: torch.nn.Module = field(repr=False)
    input: t.Any = field(repr=False)
    conf: Configuration
    time_sec: float
    accuracy_res: t.Optional[bool] = None

    @property
    def time_per_iter_ms(self) -> float:
        return self.time_sec * 1.0e3

    @property
    def qps(self) -> float:
        return self.conf.batch_size / self.time_sec

    def format(self) -> str:
        return (
            f"== Benchmark Result for: {self.conf}\n"
            f"BS: {self.conf.batch_size}, "
            f"Time per iter: {self.time_per_iter_ms:.2f}ms, "
            f"QPS: {self.qps:.2f}, "
            f"Accuracy: {self.accuracy_res} (rtol={self.conf.accuracy_rtol})"
        )


@torch.inference_mode()
def benchmark(
    model,
    inputs,
    batch_iter: int,
    batch_size: int,
) -> None:
    """
    Run fx2trt lowering and benchmark the given model according to the
    specified benchmark configuration. Prints the benchmark result for each
    configuration at the end of the run.
    """

    model = model.cuda().eval()
    inputs = [x.cuda() for x in inputs]

    # benchmark base configuration
    conf = Configuration(batch_iter=batch_iter, batch_size=batch_size)

    configurations = [
        # FP32
        replace(
            conf,
            name="TRT FP32 Eager",
            trt=True,
            torchdynamo=False,
            fp16=False,
            accuracy_rtol=1e-3,
        ),
        # torchdynamo fp16
        replace(
            conf,
            name="TRT FP16 Eager",
            trt=False,
            torchdynamo=True,
            fp16=True,
            accuracy_rtol=1e-2,
        ),
        # FP16
        replace(
            conf,
            name="torchdynamo-TRT FP32 Eager",
            trt=False,
            torchdynamo=True,
            fp16=False,
            accuracy_rtol=1e-2,
        ),
        # torchdynamo fp16
        replace(
            conf,
            name="torchdynamo-TRT FP16 Eager",
            trt=False,
            torchdynamo=True,
            fp16=True,
            accuracy_rtol=1e-2,
        ),
    ]

    results = [
        run_configuration_benchmark(deepcopy(model), inputs, conf_)
        for conf_ in configurations
    ]

    for res in results:
        print(res.format())


def benchmark_torch_function(iters: int, f, *args) -> float:
    """Estimates the average time duration for a single inference call in second

    If the input is batched, then the estimation is for the batches inference call.

    Args:
        iters: number of inference iterations to run
        f: a function to perform a single inference call

    Returns:
        estimated average time duration in second for a single inference call
    """
    with torch.inference_mode():
        f(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print("== Start benchmark iterations")
    with torch.inference_mode():
        start_event.record()
        for _ in range(iters):
            f(*args)
        end_event.record()
    torch.cuda.synchronize()
    print("== End benchmark iterations")
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def run_configuration_benchmark(
    module,
    input,
    conf: Configuration,
) -> Result:
    """
    Runs `module` through lowering logic and benchmark the module before and
    after lowering.
    """
    print(f"=== Running benchmark for: {conf}", "green")
    time = -1.0

    if conf.fp16:
        module = module.half()
        input = [i.half() for i in input]

    if conf.trt:
        # Run lowering eager mode benchmark
        lowered_module = compile(
            module,
            input,
            max_batch_size=conf.batch_size,
            lower_precision=LowerPrecision.FP16 if conf.fp16 else LowerPrecision.FP32,
        )

        time = benchmark_torch_function(conf.batch_iter, lambda: lowered_module(*input))
    elif conf.torchdynamo:
        if conf.fp16:
            optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler_fp16)
        else:
            optimize_ctx = torchdynamo.optimize(backends.fx2trt_compiler)
        with optimize_ctx:
            time = benchmark_torch_function(conf.batch_iter, module, *input)
    else:
        print("Lowering mode is not available!", "red")

    result = Result(module=module, input=input, conf=conf, time_sec=time)
    return result


if __name__ == "__main__":
    test_model = torchvision.models.resnet18()
    input = [torch.cuda.FloatTensor(64, 3, 224, 224)]  # type: ignore[attr-defined]
    benchmark(test_model, input, 100, 64)
