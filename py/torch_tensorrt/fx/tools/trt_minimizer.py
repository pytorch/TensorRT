import logging
from typing import Any, Callable, Tuple

import torch
import torch.fx.passes.net_min_base as net_min_base
from torch.fx.passes.tools_common import Tensors

from .. import InputTensorSpec, TRTInterpreter, TRTModule

_LOGGER: logging.Logger = logging.getLogger(__name__)


def lower_mod_default(
    mod: torch.fx.GraphModule,
    inputs: Tensors,
    batch_size: Any = 2048,
    use_experimental_rt: bool = False,
) -> TRTModule:
    interp = TRTInterpreter(
        mod, InputTensorSpec.from_tensors(inputs), explicit_batch_dimension=True
    )
    interpreter_result = interp.run(max_batch_size=batch_size)
    if use_experimental_rt:
        import io
        from torch_tensorrt._TRTModule import TRTModule as TRTModuleNext
        from torch_tensorrt._Device import Device

        with io.BytesIO() as engine_bytes:
            engine_bytes.write(interpreter_result.engine.serialize())
            engine_str = engine_bytes.getvalue()

        res_mod = TRTModuleNext(
            name=str(type(mod)),
            serialized_engine=engine_str,
            input_binding_names=interpreter_result.input_names,
            output_binding_names=interpreter_result.output_names,
            target_device=Device(f"cuda:{torch.cuda.current_device()}"),
            # cuda_graph_batch_size=lower_setting.cuda_graph_batch_size, # NOTE: Not sure what this is supposed to do
        )
    else:
        res_mod = TRTModule(
            interpreter_result.engine,
            interpreter_result.input_names,
            interpreter_result.output_names,
        )
    return res_mod


class TensorRTMinizerSetting(net_min_base._MinimizerSettingBase):
    def __init__(
        self, explicit_batch_dimension: Any = True, use_experiemental_rt: bool = False
    ):
        self.explicit_batch_dimension = explicit_batch_dimension
        self.use_experimental_rt = use_experiemental_rt
        super(TensorRTMinizerSetting, self).__init__()


class TensorRTMinimizer(net_min_base._MinimizerBase):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[[Any, Any, Any], Tuple[float, bool]],
        settings: TensorRTMinizerSetting = TensorRTMinizerSetting(),
        max_batch_size: Any = 2048,
        lower_fn: Callable[
            [torch.fx.GraphModule, Tensors, Any, bool], TRTModule
        ] = lower_mod_default,
    ):
        self.lower_fn = lower_fn
        self.max_batch_size = max_batch_size
        self.use_experiemental_rt = settings.use_experimental_rt
        super().__init__(module, sample_input, compare_fn, settings)

    def run_a(self, mod, inputs):
        mod.eval()
        with torch.no_grad():
            return mod(*inputs)

    def run_b(self, mod, inputs):
        mod.eval()
        try:
            mod = self.lower_fn(
                mod, inputs, self.max_batch_size, self.use_experiemental_rt
            )
            output = mod(*inputs)
        except RuntimeError as e:
            raise net_min_base.FxNetMinimizerRunFuncError(
                f"Encounter an error when processing \n{mod.graph}\n {e}"
            )
        else:
            return output

    def get_nodes(self, start=None, end=None, enable_print=False):
        nodes = self._collect_nodes(start, end)
        if enable_print:
            _LOGGER.info(f"Nodes fetched from start {start} to end {end} as: {nodes}")
        return nodes
