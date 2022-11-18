from operator import truediv
from typing import Any, List, Sequence, Tuple

import torch
from torch_tensorrt import _C
from torch_tensorrt._Device import Device


class TRTModule(torch.nn.Module):
    """TRTModule is a PyTorch module which encompasses an arbitrary TensorRT Engine.

    This module is backed by the Torch-TensorRT runtime and is fully compatibile with both
    FX / Python deployments (just ``import torch_tensorrt`` as part of the application) as
    well as TorchScript / C++ deployments since TRTModule can be passed to ``torch.jit.trace``
    and then saved.

    The forward function is simpily forward(*args: torch.Tensor) -> Tuple[torch.Tensor] where
    the internal implementation is ``return Tuple(torch.ops.tensorrt.execute_engine(list(inputs), self.engine))``

    Attributes:
        name (str): Name of module (for easier debugging)
        engine (torch.classess.tensorrt.Engine): Torch-TensorRT TensorRT Engine instance, manages [de]serialization, device configuration, profiling
        input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
        output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned
    """

    def __init__(
        self,
        name: str = "",
        serialized_engine: bytearray = bytearray(),
        input_binding_names: List[str] = [],
        output_binding_names: List[str] = [],
        target_device: Device = Device._current_device(),
    ):
        """__init__ method for torch_tensorrt.TRTModule

        Takes a name, target device, serialized TensorRT engine, and binding names / order and constructs
        a PyTorch ``torch.nn.Module`` around it.

        Args:
            name (str): Name for module
            serialized_engine (bytearray): Serialized TensorRT engine in the form of a bytearray
            input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
            output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned
            target_device: (torch_tensorrt.Device): Device to instantiate TensorRT engine on. Must be a compatible device i.e. same GPU model / compute capability as was used to build the engine

        Example:

            ..code-block::python

                with io.BytesIO() as engine_bytes:
                    engine_bytes.write(trt_engine.serialize())
                    engine_str = engine_bytes.getvalue()

                trt_module = TRTModule(
                    engine_name="my_engine",
                    serialized_engine=engine_str,
                    input_names=["x"],
                    output_names=["output"],
                )

        """

        super(TRTModule, self).__init__()
        self.input_binding_names = input_binding_names
        self.output_binding_names = output_binding_names
        self.name = name

        if serialized_engine != bytearray():
            self.engine = torch.classes.tensorrt.Engine(
                [
                    torch.ops.tensorrt.ABI_VERSION(),
                    self.name + "_engine" if self.name != "" else "tensorrt_engine",
                    target_device._to_serialized_rt_device(),
                    serialized_engine,
                    TRTModule._pack_binding_names(self.input_binding_names),
                    TRTModule._pack_binding_names(self.output_binding_names),
                ]
            )
        else:
            self.engine = None

    def get_extra_state(self):
        return (
            self.name,
            self.engine.__getstate__() if self.engine is not None else None,
            self.input_binding_names,
            self.output_binding_names,
        )

    def set_extra_state(self, state):
        self.name = state[0]
        if state[1] is not None:
            serialized_engine_info = state[1][0]
            import base64

            serialized_engine = base64.b64decode(serialized_engine_info[3])
            self.engine = torch.classes.tensorrt.Engine(
                [
                    serialized_engine_info[0],
                    serialized_engine_info[1],
                    serialized_engine_info[2],
                    serialized_engine,
                    serialized_engine_info[4],
                    serialized_engine_info[5],
                ]
            )
        else:
            self.engine = None

        self.input_binding_names = state[2]
        self.output_binding_names = state[3]

    def forward(self, *inputs):
        """Implementation of the forward pass for a TensorRT engine

        Args:
            *inputs (torch.Tensor): Inputs to the forward function, must all be ``torch.Tensor``

        Returns:
            torch.Tensor or Tuple(torch.Tensor): Result of the engine computation
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initalized yet.")

        assert len(inputs) == len(
            self.input_binding_names
        ), f"Wrong number of inputs, expected {len(self.input_binding_names)} got {len(inputs)}."

        types = [issubclass(type(i), torch.Tensor) for i in inputs]

        try:
            assert all(types)
        except:

            def is_non_tensor(i: Tuple[Any, bool]) -> bool:
                return not i[1]

            non_tensors = [i[0] for i in filter(zip(inputs, types), is_non_tensor)]
            raise RuntimeError(
                f"TRTModule expects a flattened list of tensors as input, found non tensors: {non_tensors}"
            )

        outputs = torch.ops.tensorrt.execute_engine(list(inputs), self.engine)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def enable_profiling(self, profiling_results_dir: str = None):
        """Enable the profiler to collect latency information about the execution of the engine

        Keyword Arguments:
            profiling_results_dir (str): Absolute path to the directory to sort results of profiling.
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initalized yet.")

        if profiling_results_dir is not None:
            self.engine.profile_path_prefix = profiling_results_dir
        self.engine.enable_profiling()

    def disable_profiling(self):
        """Disable the profiler"""
        if self.engine is None:
            raise RuntimeError("Engine has not been initalized yet.")

        self.engine.disable_profiling()

    def get_layer_info(self) -> str:
        """Get a JSON string containing the layer information encoded by the TensorRT engine in this module

        Returns:

            str: A JSON string which contains the layer information of the engine incapsulated in this module
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initalized yet.")

        return self.engine.get_engine_layer_info()

    def dump_layer_info(self):
        """Dump layer information encoded by the TensorRT engine in this module to STDOUT"""
        if self.engine is None:
            raise RuntimeError("Engine has not been initalized yet.")

        return self.engine.dump_engine_layer_info()

    @staticmethod
    def _pack_binding_names(binding_names: List[str]) -> str:
        delim = torch.ops.tensorrt.SERIALIZED_ENGINE_BINDING_DELIM()[0]
        return delim.join(binding_names)
