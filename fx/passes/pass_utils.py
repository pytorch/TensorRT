from typing import List, Any, Callable
from torch import fx
import logging
import torch
import tempfile
from functools import wraps
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


def validate_inference(rtol=None, atol=None):
    def _validate_inference(pass_: PassFunc) -> PassFunc:
        """
        Wraps a pass function to validate that its inference results before and
        after the pass run should be `allclose`.
        """

        @wraps(pass_)
        def pass_with_validation(
            module: fx.GraphModule, input: Input
        ) -> fx.GraphModule:
            res0 = module(*input)
            module = pass_(module, input)
            res1 = module(*input)

            tensor_res_0 = _collect_tensors(res0)
            tensor_res_1 = _collect_tensors(res1)

            for kk, (x, y) in enumerate(zip(tensor_res_0, tensor_res_1)):
                kwargs = {}
                if rtol:
                    kwargs["rtol"] = rtol
                if atol:
                    kwargs["atol"] = atol
                assert torch.allclose(
                    x, y, **kwargs
                ), f"pass {pass_} failed correctness check due to output {kk}"
            return module

        return pass_with_validation

    return _validate_inference


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
            print(f"== Log pass {pass_} before/after graph to {f.name}")
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
