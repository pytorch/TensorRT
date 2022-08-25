import logging
import tempfile
from functools import wraps
from typing import Any, Callable, List

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any
_LOGGER: logging.Logger = logging.getLogger(__name__)

PassFunc = Callable[[fx.GraphModule, Input], fx.GraphModule]


def chain_passes(*passes: PassFunc) -> PassFunc:
    """
    Chains a sequence of pass functions to form a single pass function
    """

    def parent_pass(module: fx.GraphModule, input: Input) -> fx.GraphModule:
        for pass_ in passes:
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module, input)
        return module

    return parent_pass


# (TODO(shirongwu): Add exception notification for fblearner flow when available, notify oncall
# on pass that failed accuracy check.
def validate_inference(rtol=None, atol=None, suppress_accuracy_check_failure=False):
    def _validate_inference(pass_: PassFunc) -> PassFunc:
        """
        Wraps a pass function to validate that its inference results before and
        after the pass run should be `allclose`.
        """

        @wraps(pass_)
        def pass_with_validation(
            module: fx.GraphModule,
            input: Input,
            *args,
            **kwargs,
        ) -> fx.GraphModule:
            res0 = module(*input)
            processed_module = pass_(module, input, *args, **kwargs)
            res1 = processed_module(*input)

            tensor_res_0 = _collect_tensors(res0)
            tensor_res_1 = _collect_tensors(res1)

            for kk, (x, y) in enumerate(zip(tensor_res_0, tensor_res_1)):
                kwargs = {"equal_nan": True}
                if rtol:
                    kwargs["rtol"] = rtol
                if atol:
                    kwargs["atol"] = atol
                # If tensors are on different devices, make sure to compare
                # their copies that are on the same device.
                if x.get_device() != y.get_device():
                    x = x.cpu()
                    y = y.cpu()
                accuracy_check = torch.allclose(x, y, **kwargs)
                if not accuracy_check:
                    _LOGGER.error(
                        f"Pass {pass_} failed correctness check, get original model output as {x} and processed model output as {y} for output {kk}."
                    )
                    if suppress_accuracy_check_failure:
                        _LOGGER.error(
                            f"Pass {pass_} failed correctness check due to output {kk}."
                        )
                        return processed_module
                    else:
                        raise RuntimeError(
                            f"Pass {pass_} failed correctness check due to output {kk}"
                        )
            return processed_module

        return pass_with_validation

    return _validate_inference


Decorator = Callable[[Callable], Callable]


def decorate_method(dec_for_function: Decorator) -> Decorator:
    def dec_for_method(unbounded_method) -> Callable:
        def decorated_unbounded_method(self, *args, **kwargs):
            @dec_for_function
            def bounded_method(*args, **kwargs):
                return unbounded_method(self, *args, **kwargs)

            return bounded_method(*args, **kwargs)

        return decorated_unbounded_method

    return dec_for_method


def log_perf_before_after(pass_: PassFunc) -> PassFunc:
    """
    Wraps a pass function to log perf of the module before and after the pass
    """

    @wraps(pass_)
    def check_perf_with_before_after_log(
        module: fx.GraphModule, input: Input
    ) -> fx.GraphModule:
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
            # print("== Start benchmark iterations")
            with torch.inference_mode():
                start_event.record()
                for _ in range(iters):
                    f(*args)
                end_event.record()
            torch.cuda.synchronize()
            # print("== End benchmark iterations")
            return (start_event.elapsed_time(end_event) * 1.0e-3) / iters

        time_before = benchmark_torch_function(100, lambda: module(*input))
        _LOGGER.info(f"[{pass_}] Perf Before(eager mode): {time_before}")

        module = pass_(module, input)
        time_after = benchmark_torch_function(100, lambda: module(*input))
        _LOGGER.info(f"[{pass_}] Perf After(eager mode): {time_after}")
        return module

    return check_perf_with_before_after_log


def log_before_after(pass_: PassFunc) -> PassFunc:
    """
    Wraps a pass function to log the module graph before and after the pass
    """

    @wraps(pass_)
    def pass_with_before_after_log(
        module: fx.GraphModule, input: Input
    ) -> fx.GraphModule:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as f:
            _LOGGER.info(f"== Log pass {pass_} before/after graph to {f.name}")
            print(f"[{pass_}] Before:\n{module.graph}", file=f)
            module = pass_(module, input)
            print(f"[{pass_}] After:\n{module.graph}", file=f)
            return module

    return pass_with_before_after_log


def _collect_tensors(arg: fx.node.Argument) -> List[torch.Tensor]:
    """Collects all the tensors found in a nested container object"""
    res: List[torch.Tensor] = []

    def collect(x: fx.node.Argument) -> fx.node.Argument:
        if isinstance(x, torch.Tensor):
            res.append(x)
        return x

    fx.node.map_aggregate(arg, collect)
    return res
