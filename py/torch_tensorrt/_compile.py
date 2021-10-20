from typing import List, Dict, Any
from torch_tensorrt import _enums
import torch_tensorrt.ts
from torch_tensorrt import logging
import torch
from enum import Enum

class _IRType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout
    """
    ts = 0
    fx = 1

def _module_ir(module: Any, ir: str) -> _IRType.ts:
	# Possible module types
	module_is_tsable = any(isinstance(module, t) for t in [torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction])
	module_is_fxable = any(isinstance(module, t) for t in [torch.nn.Module, torch.fx.GraphModule])

	ir_targets_torchscript = any([ir == opt for opt in ["torchscript", "ts"]])
	ir_targets_fx = ir == "fx"

	if module_is_tsable and ir_targets_torchscript:
		return _IRType.ts
	elif module_is_fxable and ir_targets_fx:
		if isinstance(module, torch.fx.GraphModule):
			raise ValueError("Was given a torch.fx.GraphModule, fx is not currently supported by Torch-TensorRT")
		elif ir_targets_fx:
			raise ValueError("Preferred ir was set to \"fx\" which is currently not supported by Torch-TensorRT")
		else:
			raise ValueError("Torch-TensorRT currently does not support fx")
		# return _IRType.fx
	else:
			if ir == "default":
				# Options are listed in order of preference
				if module_is_tsable:
					logging.log(logging.Level.Info, "ir was set to default, using TorchScript as ir")
					return _IRType.ts
				elif module_is_fxable:
					raise ValueError("Was given a torch.fx.GraphModule, fx is not currently supported by Torch-TensorRT")
					#logging.log(logging.Level.Info, "ir was set to default, using TorchScript as fx")
					#return _IRType.fx
				else:
					raise ValueError("Module was provided with in an unsupported format")
			else:
				raise ValueError("Unknown ir was requested")

def compile(module: Any,
						ir="default",
						inputs=[],
						enabled_precisions=set([_enums.dtype.float]),
						**kwargs):

	target_ir = _module_ir(module, ir)
	if target_ir == _IRType.ts:
		ts_mod = module
		if isinstance(module, torch.nn.Module):
			logging.log("Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript")
			ts_mod = torch.jit.script(module)
		return torch_tensorrt.ts.compile(ts_mod, inputs=inputs, enabled_precisions=enabled_precisions, **kwargs)
	elif target_ir == _IRType.fx:
		raise RuntimeError("fx is currently not supported")
	else:
		raise RuntimeError("Module is an unknown format or the ir requested is unknown")

def convert_method_to_trt_engine(module: Any,
																 method_name: str,
																 ir="default",
																 inputs=[],
																 enabled_precisions=set([_enums.dtype.float]),
																 **kwargs):
	target_ir = _module_ir(module, ir)
	if target_ir == _IRType.ts:
		ts_mod = module
		if isinstance(module, torch.nn.Module):
			logging.log("Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript")
			ts_mod = torch.jit.script(module)
		return torch_tensorrt.ts.convert_method_to_trt_engine(ts_mod, method_name, inputs=inputs, enabled_precisions=enabled_precisions, **kwargs)
	elif target_ir == _IRType.fx:
		raise RuntimeError("fx is currently not supported")
	else:
		raise RuntimeError("Module is an unknown format or the ir requested is unknown")