from typing import Callable, Any, Sequence, NamedTuple
from torch.fx.passes.pass_manager import PassManager
from .lower_basic_pass import run_const_fold
from fx2trt_oss.fx.lower_setting import LowerSetting
from functools import wraps
import torch
from torch.fx.passes.shape_prop import ShapeProp

Input = Sequence[Any]


class LowerPassContext(NamedTuple):
    """
    Args:
    input: module input

    lower_setting: lower setting

    trace_func: desired graph trace function for lowering
    """
    input: Input
    lower_setting: "LowerSetting"
    trace_func: Callable

def wrapper(fn: Callable, input) -> Callable:
    @wraps(fn)
    def wrapped_fn(gm):
        if isinstance(gm, torch.fx.GraphModule):
            ShapeProp(gm).propagate(*input)
        return fn(gm, input)

    return wrapped_fn

class LowerPassManagerBuilder:
    """
    Build PassManager for lowering
    """
    def __init__(self, lower_pass_context:LowerPassContext):
        self._build_context = lower_pass_context

    def _const_fold_pass(self) -> PassManager:
        passes = [
            wrapper(self._build_context.trace_func, self._build_context.input),
            run_const_fold,
        ]
        return PassManager.build_from_passlist(passes)


    def graph_optimization_pass(self) -> PassManager:
        passes = [
            wrapper(self._build_context.trace_func, self._build_context.input),
        ]
        for p in self._build_context.lower_setting.customized_fuse_pass:
            passes.append(wrapper(p, self._build_context.input))
        return PassManager.build_from_passlist(passes)


    def build_lower_pipeline(self) -> PassManager:
        passes = []

        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm
