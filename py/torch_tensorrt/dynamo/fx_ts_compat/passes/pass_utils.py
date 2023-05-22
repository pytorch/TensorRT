import io
import logging
import tempfile
from datetime import datetime
from functools import wraps
from typing import Any, Callable, List, Optional

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp
from torch_tensorrt import _Input

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any
_LOGGER: logging.Logger = logging.getLogger(__name__)

PassFunc = Callable[[fx.GraphModule, Input], fx.GraphModule]

RELAX_ACCURACY_FAILURE: bool = False
FINAL_CHECK_ATOL_MULTIPLIER: float = 10
FINAL_CHECK_RTOL_MULTIPLIER: float = 10


def extract_example_tensors_from_input(
    inputs: Any, device: torch.device = torch.device("cuda")
):
    input_tensors = []
    for input_obj in inputs:
        if isinstance(input_obj, _Input.Input):
            if isinstance(input_obj.shape, dict):
                input_tensors.append(
                    input_obj.example_tensor(optimization_profile_field="opt_shape").to(
                        device
                    )
                )
            else:
                input_tensors.append(input_obj.example_tensor().to(device))
        elif isinstance(input_obj, torch.Tensor):
            input_tensors.append(input_obj)
        else:
            raise ValueError(
                "Invalid input type provided in the FX lowering. Expected type: torch_tensorrt.Input or torch.Tensor"
            )

    return input_tensors


class RelaxAccuracyCheckMode:
    """
    Basically a context manager that controls a global variable that controls
    the accuracy check mode. Use it like
    with RelaxAccuracyCheckMode(True):
        fx2trt()
    """

    def __init__(
        self,
        mode: bool,
        final_atol_multiplier: Optional[float] = None,
        final_rtol_multiplier: Optional[float] = None,
    ):
        """
        Arguments:
        mode: whether we relax the immediate accuracy check failure or not. If yes, we will do an extra
        accruacy check by raising the tolerance by the multipler times and only raise error if that fails.
        This is to avoid catastrophic errors.
        final_atol_multiplier [optional]: set FINAL_CHECK_ATOL_MULTIPLIER if specifier.
        final_rtol_multiplier [optional]: set FINAL_CHECK_RTOL_MULTIPLIER if specifier.
        """
        global RELAX_ACCURACY_FAILURE
        global FINAL_CHECK_ATOL_MULTIPLIER
        global FINAL_CHECK_RTOL_MULTIPLIER
        self._old_mode = (
            RELAX_ACCURACY_FAILURE,
            FINAL_CHECK_ATOL_MULTIPLIER,
            FINAL_CHECK_RTOL_MULTIPLIER,
        )
        RELAX_ACCURACY_FAILURE = mode
        FINAL_CHECK_ATOL_MULTIPLIER = (
            final_atol_multiplier
            if final_atol_multiplier
            else FINAL_CHECK_ATOL_MULTIPLIER
        )
        FINAL_CHECK_RTOL_MULTIPLIER = (
            final_rtol_multiplier
            if final_rtol_multiplier
            else FINAL_CHECK_RTOL_MULTIPLIER
        )
        _LOGGER.info(
            f"Set new relaxed accuracy check mode: {RELAX_ACCURACY_FAILURE=}, {FINAL_CHECK_ATOL_MULTIPLIER=}, {FINAL_CHECK_RTOL_MULTIPLIER=}"
        )

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        global RELAX_ACCURACY_FAILURE
        global FINAL_CHECK_ATOL_MULTIPLIER
        global FINAL_CHECK_RTOL_MULTIPLIER
        (
            RELAX_ACCURACY_FAILURE,
            FINAL_CHECK_ATOL_MULTIPLIER,
            FINAL_CHECK_RTOL_MULTIPLIER,
        ) = self._old_mode
        _LOGGER.info(
            f"Restored old relaxed accuracy check mode: {RELAX_ACCURACY_FAILURE=}, {FINAL_CHECK_ATOL_MULTIPLIER=}, {FINAL_CHECK_RTOL_MULTIPLIER=}"
        )


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
def validate_inference(
    rtol=None,
    atol=None,
    device=torch.device(torch.cuda.current_device()),
    suppress_accuracy_check_failure=True,
):
    def _validate_inference(pass_: PassFunc) -> PassFunc:
        """
        Wraps a pass function to validate that its inference results before and
        after the pass run should be `close`.
        """

        @wraps(pass_)
        def pass_with_validation(
            module: fx.GraphModule,
            input: Input,
            *args,
            **kwargs,
        ) -> fx.GraphModule:
            if suppress_accuracy_check_failure:
                return pass_(module, input, *args, **kwargs)
            else:
                input_tensors = extract_example_tensors_from_input(input, device)
                res0 = module(*input_tensors)
                processed_module = pass_(module, input, *args, **kwargs)
                res1 = processed_module(*input_tensors)
                tensor_res_0 = _collect_tensors(res0)
                tensor_res_1 = _collect_tensors(res1)
                relax_accuracy_check_failure = RELAX_ACCURACY_FAILURE

                for kk, (x, y) in enumerate(zip(tensor_res_0, tensor_res_1)):
                    kwargs2 = {"equal_nan": True}
                    if rtol:
                        kwargs2["rtol"] = rtol
                    if atol:
                        kwargs2["atol"] = atol
                    kwargs2[
                        "msg"
                    ] = (
                        lambda msg: f"Pass {pass_} failed correctness check due at output {kk}:\n{msg}"
                    )
                    # If tensors are on different devices, make sure to compare
                    # their copies that are on the same device.
                    if x.get_device() != y.get_device():
                        x = x.cpu()
                        y = y.cpu()
                    try:
                        torch.testing.assert_close(x, y, **kwargs2)
                    except Exception as e:
                        if relax_accuracy_check_failure:
                            _LOGGER.error(f"{e}")
                            kwargs2["rtol"] *= FINAL_CHECK_RTOL_MULTIPLIER
                            kwargs2["atol"] *= FINAL_CHECK_ATOL_MULTIPLIER
                            new_atol = kwargs2["atol"]
                            new_rtol = kwargs2["rtol"]
                            _LOGGER.info(
                                f"Do a sanity check to see whether things are completely wrong with {new_atol=}, {new_rtol=}"
                            )
                            torch.testing.assert_close(x, y, **kwargs2)
                            return processed_module
                        else:
                            raise e

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
        before_io = io.StringIO()
        after_io = io.StringIO()
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as f:
            print(f"[{pass_}] Before:\n{module.graph}", file=f)
            print(module.graph, file=before_io)
            start_time = datetime.now()
            module = pass_(module, input)
            t_elapsed = datetime.now() - start_time
            print(f"[{pass_}] After:\n{module.graph}", file=f)
            print(module.graph, file=after_io)
            t = before_io.getvalue() == after_io.getvalue()
            _LOGGER.info(
                f"== Log pass {pass_} before/after graph to {f.name}, before/after are the same = {t}, time elapsed = {t_elapsed}"
            )
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
