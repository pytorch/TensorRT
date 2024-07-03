import collections
import pickle
from typing import Any
import torch

from torch_tensorrt.dynamo._refit import refit_module_weights
import torch_tensorrt as torch_trt
import torch
from torch_tensorrt.dynamo.utils import prepare_inputs
from torch_tensorrt.dynamo._tracer import trace as dynamo_trace
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile

class RefitFlag:
    def __init__(self):
        self.flag = False

    def set_on(self):
        self.flag = True
        print("RefitFlag is set to ON.")

    def set_off(self):
        self.flag = False
        print("RefitFlag is set to OFF.")

class MutableTorchTensorRTModule(object):
    def __init__(self, pytorch_model, sample_inputs, enabled_precisions_set, **kwargs) -> None:
        self.refit_flag = RefitFlag()
        self.original_inputs = sample_inputs
        if not isinstance(sample_inputs, collections.abc.Sequence):
            sample_inputs = [sample_inputs]
        self.sample_inputs = tuple(sample_inputs)
        self.torchtrt_inputs = prepare_inputs(self.sample_inputs)
        self.kwargs = kwargs
        self.original_model = pytorch_model
        self.pytorch_model = _make_refit_change_trigger(pytorch_model, self.refit_flag)
        self.gm = None
        self.enabled_precisions_set = enabled_precisions_set

    def load_state_dict(self, sd):
        self.refit_flag.set_on()
        self.pytorch_model.load_state_dict(sd)

    def refit_gm(self):
        if self.exp_program is None:
            self.exp_program = torch.export.export(self.pytorch_model, self.sample_inputs)
        # TODO: Check refit condition and fallback to recompile
        self.exp_program._state_dict = MutableTorchTensorRTModule._transform_state_dict(self.pytorch_model.state_dict())
        self.gm = refit_module_weights(self.gm, self.exp_program, self.sample_inputs)
        

    def compile(self):
        
        # Export the module
        self.exp_program = dynamo_trace(self.original_model, self.torchtrt_inputs, **self.kwargs)
        self.gm = dynamo_compile(
            self.exp_program,
            inputs=self.torchtrt_inputs,
            enabled_precisions=self.enabled_precisions_set,
            make_refitable=True,
            **self.kwargs,
        )
        
    def _transform_state_dict(sd):
        return {k: torch.nn.Parameter(v, requires_grad=False) for k, v in sd.items()}
        
    def __getattr__(self, name):
    
        if name in self.__dict__:
            # this object has it
            return getattr(self, name)

        return getattr(self.pytorch_model, name)
        
        # raise AttributeError(f"'{type(self.pytorch_model)}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        # We can update this once the kwarg pull request got merged
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        # TODO: Check the inputs is the same as the sample input
        if self.refit_flag.flag:
            print("Model weight change detected. Refitting the module...")
            self.refit_flag.set_off()
            self.refit_gm()

        return self.gm(*args, **kwargs)
    
    


def _make_refit_change_trigger(obj: Any, refit_flag: RefitFlag) -> Any:

    class ChangeTriggerWrapper(obj.__class__):
        def __init__(self, obj: Any):
            object.__setattr__(self, 'instance', obj)

        def __getattr__(self, name: str):
            # This will cause infinte loop if there is a cycle
            obj = getattr(self.instance, name)
            if not hasattr(obj, '__dict__'):
                return obj 
            else:
                return _make_refit_change_trigger(obj, refit_flag)
            
        def __setattr__(self, name:str, value: Any):
            self._on_change()
            setattr(self.instance, name, value)

        def __delattr__(self, name: str):
            self._on_change()
            delattr(self.instance, name, )

        def _on_change(self):
            refit_flag.set_on()
            print("Change!")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            print("Warning: uncatched change in function!")
            self.instance(*args, **kwargs)

    return ChangeTriggerWrapper(obj)

