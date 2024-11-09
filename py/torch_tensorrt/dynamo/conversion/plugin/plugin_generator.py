import tensorrt as trt
import cupy as cp
import torch
import numpy as np

import logging


from enum import IntEnum
from typing import List


logger = logging.getLogger("CustomPlugin")

_numpy_to_plugin_field_type = {
    np.dtype('int32'): trt.PluginFieldType.INT32,
    np.dtype('int16'): trt.PluginFieldType.INT16,
    np.dtype('int8'): trt.PluginFieldType.INT8,
    np.dtype('bool'): trt.PluginFieldType.INT8,
    np.dtype('int64'): trt.PluginFieldType.INT64,
    np.dtype('float32'): trt.PluginFieldType.FLOAT32,
    np.dtype('float64'): trt.PluginFieldType.FLOAT64,
    np.dtype('float16'): trt.PluginFieldType.FLOAT16
}

_built_in_to_plugin_field_type = {
    int: trt.PluginFieldType.INT64,
    float: trt.PluginFieldType.FLOAT64,
    bool: trt.PluginFieldType.INT8,
    # str is handled separately, so not needed here
}

class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2

class CustomPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):  # type: ignore[misc]
    def __init__(
        self, plugin_name : str, attrs, phase = None
    ):
        # TODO: needs an additional passed in arguments to specify the needs for each plugin
        # such as the one here: https://github.com/NVIDIA/TensorRT/blob/40efe7e9f2492657bbc455c4e2876e2ec792b812/samples/python/python_plugin/circ_pad_plugin_multi_tactic.py#L83
        trt.IPluginV3.__init__(self)
        # Core capability, plugin attributes and behaviors common to both the build and runtime phases of a plugin’s lifetime
        trt.IPluginV3OneCore.__init__(self)
        # Build capability, plugin attributes and behaviors that the plugin must exhibit for the TensorRT builder.
        trt.IPluginV3OneBuild.__init__(self)
        # Runtime capability, plugin attributes and behaviors that the plugin must exhibit for it to be executable
        trt.IPluginV3OneRuntime.__init__(self)       
        
        # <ANY NON TENSOR INPUTS SHOULD BE AN ATTRIBUTE OF THE PLUGIN>
        # setattr(<name of input>, <default value for that type>) 
        # self.pads = []
        # self.X_shape: List[int] = []
 
        self.num_outputs = 1 # Defined by schema 
        self.plugin_namespace = ""
        self.plugin_name = plugin_name
        self.plugin_version = "1"   

        # Set the timing cache ID to prevent unnecessary timing of second plugin instance
        self.timing_cache_id = ""

        self.attrs = attrs
        
        self.tactic = None
        

        # <GENERATE CODE FOR TAKING A FIELD COLLECTION CONTAINING THE NON TENSOR INPUTS AND SETTING AN ATTR> 
        # ex.
        # TODO: need to parse the field collection here
        # if fc is not None:
        #     assert fc[0].name == "pads"
        #     self.pads = fc[0].data

        if phase is not None:
            self.phase = phase

    def get_capability_interface(self, type):
        return self

    def get_output_data_types(
        self, input_types: List[trt.DataType]
    ) -> trt.DataType:
        # WE CAN USE THE FAKE TENSOR IMPLEMENTATION TO FIGURE OUT THE EXPECTED OUTPUT DATA TYPE 
        # with torch.fake_tensor():
        #      <GENERATE FAKE INPUTS OF TYPE INPUT_TYPES>
        #      fake_outputs = torch.ops.<custom_ns>.<custom_op>(*fake_inputs)

        # return fake_outputs[index]

        # The example case here is simple for experiment
        return [input_types[0]]

    def get_output_shapes(
        self,
        inputs: List[trt.DimsExprs],
        shape_inputs,
        exprBuilder: trt.IExprBuilder,
    ) -> trt.DimsExprs:
        
        print(inputs)

    #    WE NEED TO FIND A WAY TO GO FROM FAKE TENSOR IMPL TO CONSTRUCTING A DIMSEXPR 
    #    THIS IS SOLVED IN SHAPE PROP IN PYTORCH WHERE SHAPE PROP CAN GIVE SYMINTS THAT ENCODE THE 
    #    SHAPE MAP. 
        output_dims = trt.DimsExprs(inputs[0])

        return [output_dims]
    
    def get_fields_to_serialize(self):
        # should be passed in as another argument
        field_names = []

        for key, value in self.attrs.items():
            if isinstance(value, np.ndarray):
                field_names.append(
                    trt.PluginField(
                        key,
                        value,
                        _numpy_to_plugin_field_type[np.dtype(value.dtype)],
                    )
                )
            elif isinstance(value, str):
                field_names.append(
                    trt.PluginField(key, value.encode(), trt.PluginFieldType.CHAR)
                )
            elif isinstance(value, bytes):
                field_names.append(
                    trt.PluginField(key, value, trt.PluginFieldType.UNKNOWN)
                )
            else:
                field_names.append(
                    trt.PluginField(
                        key,
                        np.array([value]),
                        _built_in_to_plugin_field_type[type(value)],
                    )
                )

        return trt.PluginFieldCollection(field_names)

    def configure_plugin(self, inp, out):
        pass

    def on_shape_change(self, inp, out):
        return
        X_dims = inp[0].dims
        self.X_shape = np.zeros((len(X_dims),))
        for i in range(len(X_dims)):
            self.X_shape[i] = X_dims[i]

    def supports_format_combination(self, pos, in_out, num_inputs):
        return 
        assert num_inputs == 1
        assert pos < len(in_out)

        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF

        # output should have the same type as the input
        if pos == 1:
            return in_out[0].desc.type == desc.type

        assert False


    def enqueue(
        self,
        input_desc: List[trt.PluginTensorDesc],
        output_desc: List[trt.PluginTensorDesc],
        inputs,
        outputs,
        workspace: int,
        stream: int,
    ) -> None:
        # input and output memory handling
        input_mems = [None] * (len(inputs))

        for i in range(len(inputs)): 
            input_mems[i] = cp.cuda.UnownedMemory(inputs[i], np.prod(input_desc[i].dims) * cp.dtype(trt.nptype(input_desc[i].type)).itemsize, self)

        output_mems = [None] * (len(outputs))

        for i in range(len(outputs)):
            output_mems[i] = cp.cuda.UnownedMemory(outputs[i], np.prod(output_desc[i].dims) * cp.dtype(trt.nptype(output_desc[i].type)).itemsize, self)
    

        input_data = [None] * ((len(inputs)))
        for i in range(len(inputs)):
            input_data[i] = cp.ndarray(tuple(input_desc[i].dims), dtype=input_desc[i].type, memptr = cp.cuda.MemoryPointer(input_mems[i], 0))

        output_data = [None] * ((len(outputs)))
        for i in range(len(outputs)):
            output_data[i] = cp.ndarray((np.prod(output_desc[i].dims)), dtype = output_desc[i].type, memptr = cp.cuda.MemoryPointer(output_mems[i], 0))

        #TODO: This is just for a simple case for elementwise operations
        # using Torch implementation for now
        input_torch_0 = torch.as_tensor(input_data[0], device='cuda')
        input_torch_1 = torch.as_tensor(input_data[1], device='cuda')

        output = torch.ops.torchtrt_ex.elementwise_add(input_torch_0, input_torch_1)

        cp.copyto(output_data, output)


    def attach_to_context(self, context):
        return self.clone()
    
    def get_valid_tactics(self):
        return [int(Tactic.TORCH), int(Tactic.TRITON)]

    def set_tactic(self, tactic):
        self.tactic = Tactic(tactic)

        # if self.phase == trt.TensorRTPhase.RUNTIME:
        #     logger.info(f"Best tactic chosen: {self.tactic}")

    def clone(self):
        cloned_plugin = CustomPlugin(self.plugin_name, self.attrs)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin


class PluginCreator(trt.IPluginCreatorV3One):  # type: ignore[misc]
    def __init__(self, plugin_name : str, plugin_namespace : str, attrs):
        trt.IPluginCreatorV3One.__init__(self)  

        self.name = plugin_name
        self.plugin_namespace = plugin_namespace
        self.plugin_version = "1"
        
        field_names = []
        for name, (builtin, type_) in attrs.items():
            if builtin:
                if type_ is str:
                    field_names.append(
                        trt.PluginField(name, b"", trt.PluginFieldType.CHAR)
                    )
                elif type_ is bytes:
                    field_names.append(
                        trt.PluginField(name, b"", trt.PluginFieldType.UNKNOWN)
                    )
                else:
                    field_names.append(
                        trt.PluginField(
                            name, np.array([]), _built_in_to_plugin_field_type[type_]
                        )
                    )
            else:
                field_names.append(
                    trt.PluginField(
                        name, np.array([]), _numpy_to_plugin_field_type[np.dtype(type_)]
                    )
                )

        self.field_names = trt.PluginFieldCollection(field_names)

    def create_plugin(
        self, name: str, field_collection, phase=None
    ) -> CustomPlugin:

        
        attrs = {}
        # for f in fc:
        #     if f.name not in desc.input_attrs:
        #         raise AssertionError(
        #             f"Unexpected attribute {f.name} provided to create_plugin. Expected one of {desc.input_attrs.keys()}."
        #         )

        #     if _is_numpy_array(desc.input_attrs[f.name]):
        #         attrs[f.name] = f.data.astype(_infer_numpy_type(desc.input_attrs[f.name]))
        #     else:
        #         attrs[f.name] = desc.input_attrs[f.name](f.data)
                
        custom_plugin = CustomPlugin(name, attrs)
        
        return custom_plugin


# Looks like deserilaize required? Not found in the example here: https://github.com/NVIDIA/TensorRT/blob/main/samples/python/python_plugin/circ_pad_plugin_multi_tactic.py
    # def deserialize_plugin(self, name: str, data: bytes) -> CircularPaddingPlugin:
    #     dict = pkl.loads(data)
    #     deserialized = <PLUGIN CLASS>()
    #     deserialized.__dict__.update(dict)
    #     return deserialized

# TRT_PLUGIN_REGISTRY = trt.get_plugin_registry()
# TRT_PLUGIN_REGISTRY.register_creator(CircularPaddingPluginCreator(), "") 