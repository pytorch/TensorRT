from functools import partial, wraps
from typing import Callable, Any, Sequence, NamedTuple

import torch
from fx2trt_oss.fx.lower_setting import LowerSetting
from fx2trt_oss.fx.observer import Observer
from fx2trt_oss.fx.passes.remove_duplicate_output_args import (
    remove_duplicate_output_args,
)
from torch import nn
from torch.fx.passes.pass_manager import PassManager, inplace_wrapper
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.splitter_base import SplitResult

from .lower_basic_pass import run_const_fold

Input = Sequence[Any]


# ----------------------------------------------------------------------
# OBSERVERS
# ----------------------------------------------------------------------
# List of observers. We can subscribe to them by calling its `add(callback)`
# function from anywhere in code:
#
# >>> from fx2trt_oss.fx.lower import FUSE_PASSES_POST_OBSERVER
# >>> with FUSE_PASSES_POST_OBSERVER.add(print_module_and_input):
# >>>     # print_module_and_input will be called right after the fuse passes
# >>>     lower(module, sample_input)

# Observer for the model after the fuse passes.
FUSE_PASSES_POST_OBSERVER: Observer[Callable[[nn.Module, Input], None]] = Observer(
    "FUSE_PASSES_POST_OBSERVER"
)

# Observer for the TRT split submodules before lowering
LOWER_SPLIT_PRE_OBSERVER: Observer[Callable[[str, nn.Module, Input], None]] = Observer(
    "LOWER_SPLIT_PRE_OBSERVER"
)

# Observer for the TRT split submodules after lowering
LOWER_SPLIT_POST_OBSERVER: Observer[Callable[[str, nn.Module, Input], None]] = Observer(
    "LOWER_SPLIT_POST_OBSERVER"
)
# ----------------------------------------------------------------------


def wrapper(fn: Callable, input) -> Callable:
    @wraps(fn)
    def wrapped_fn(gm):
        if isinstance(gm, torch.fx.GraphModule):
            ShapeProp(gm).propagate(*input)
        return fn(gm, input)

    return wrapped_fn


class LowerPassManagerBuilder:
    """
    Build PassManager for lowering.

     Attributes:
        lower_setting: Setting that will be used during process of lowering, see lower_setting.py for the details.
        _trace_func: fx trace function for TRT conversion.
        _split_func: the fx2trt split function.
        _lower_func: function to create and run `TRTInterpreter` to convert `fx.GraphModule`
            into a TensorRT engine.

    """

    def __init__(
        self,
        lower_setting: LowerSetting,
        trace_func: Callable,
        split_func: Callable,
        lower_func: Callable,
    ):
        self.lower_setting = lower_setting
        self._trace_func = trace_func
        self._split_func = split_func
        self._lower_func = lower_func

    def _const_fold_pass(self) -> PassManager:
        passes = [
            wrapper(self._trace_func, self._input),
            run_const_fold,
        ]
        return PassManager.build_from_passlist(passes)

    def graph_optimization_pass(self) -> PassManager:
        passes = [
            wrapper(self._trace_func, self._input),
        ]
        for p in self.lower_setting.customized_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        for p in self.lower_setting.lower_basic_fuse_pass.passes:
            passes.append(wrapper(p, self._input))
        passes.append(
            inplace_wrapper(partial(FUSE_PASSES_POST_OBSERVER.observe, self._input))
        )

        return PassManager.build_from_passlist(passes)

    def _split_pass(self) -> PassManager:
        passes = [
            partial(
                self._split_func, inputs=self._input, lower_setting=self.lower_setting
            )
        ]
        passes.append(
            inplace_wrapper(
                lambda split_result: remove_duplicate_output_args(
                    split_result.split_module, split_result.submodule_inputs.keys()
                )
            )
        )
        return PassManager.build_from_passlist(passes)

    def _lower_pass(self) -> PassManager:
        def lower_func(split_result: SplitResult) -> nn.Module:
            for submod_name, submod_inputs in split_result.submodule_inputs.items():
                submod = getattr(split_result.split_module, submod_name)

                LOWER_SPLIT_PRE_OBSERVER.observe(submod_name, submod, submod_inputs)

                # Only acc submodules will be lowered.
                if not submod_name.startswith(split_result.non_acc_submodule_prefix):
                    lowered_module = self._lower_func(
                        submod, submod_inputs, self.lower_setting, submod_name
                    )
                    setattr(split_result.split_module, submod_name, lowered_module)
                    LOWER_SPLIT_POST_OBSERVER.observe(
                        submod_name, lowered_module, submod_inputs
                    )

            return split_result.split_module

        return PassManager.build_from_passlist([lower_func])

    def build_lower_pipeline(self, input: Input) -> PassManager:
        self._input = input
        passes = []

        passes.append(self._const_fold_pass())
        passes.append(self.graph_optimization_pass())
        passes.append(self._split_pass())
        passes.append(self._lower_pass())

        pm = PassManager.build_from_passlist(passes)
        return pm
