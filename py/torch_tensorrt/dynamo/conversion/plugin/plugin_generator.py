import tensorrt as trt



class CustomPlugin(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):  # type: ignore[misc]
    def __init__(
        self, plugin_name : str, fc = None, phase = None
    ):
        # TODO: needs an additional passed in arguments to specify the needs for each plugin
        # such as the one here: https://github.com/NVIDIA/TensorRT/blob/40efe7e9f2492657bbc455c4e2876e2ec792b812/samples/python/python_plugin/circ_pad_plugin_multi_tactic.py#L83
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)       
        
         # <ANY NON TENSOR INPUTS SHOULD BE AN ATTRIBUTE OF THE PLUGIN>
        # setattr(<name of input>, <default value for that type>) 
        # self.pads = []
        # self.X_shape: List[int] = []
 
        self.num_outputs = 1 # Defined by schema 
        self.plugin_namespace = ""
        self.plugin_name = plugin_name
        self.plugin_version = "1"

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

    def get_output_datatypes(
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
        output_index: int,
        inputs: List[trt.DimsExprs],
        exprBuilder: trt.IExprBuilder,
    ) -> trt.DimsExprs:
        

    #    WE NEED TO FIND A WAY TO GO FROM FAKE TENSOR IMPL TO CONSTRUCTING A DIMSEXPR 
    #    THIS IS SOLVED IN SHAPE PROP IN PYTORCH WHERE SHAPE PROP CAN GIVE SYMINTS THAT ENCODE THE 
    #    SHAPE MAP. 
        output_shape = trt.DimsExprs(inputs[0])

        return [output_shape]
    
    def get_fields_to_serialize(self):
        # should be passed in as another argument
        return trt.PluginFieldCollection([
            trt.PluginField("pads", self.pads, trt.PluginFieldType.INT32)
        ])

    def configure_plugin(self, inp, out):
        pass

        def on_shape_change(self, inp, out):
        X_dims = inp[0].dims
        self.X_shape = np.zeros((len(X_dims),))
        for i in range(len(X_dims)):
            self.X_shape[i] = X_dims[i]

    def supports_format_combination(self, pos, in_out, num_inputs):
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
        inputs: List[int],
        outputs: List[int],
        workspace: int,
        stream: int,
    ) -> None:
        ...

    def attach_to_context(self, context):
        return self.clone()
    
    def get_valid_tactics(self):
        return [int(Tactic.TORCH), int(Tactic.TRITON)]

    def set_tactic(self, tactic):
        self.tactic = Tactic(tactic)

        if self.phase == trt.TensorRTPhase.RUNTIME:
            logger.info(f"Best tactic chosen: {self.tactic}")

    def clone(self) -> Self:
        # 


class PluginCreator(trt.IPluginCreatorV3One):  # type: ignore[misc]
    def __init__(self, plugin_name : str, plugin_field_names : trt.PluginFieldCollection):
        super().__init__()

        self.name = plugin_name
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = plugin_field_names

    def create_plugin(
        self, name: str, field_collection: trt.PluginFieldCollection_
    ) -> CustomPlugin:
        return CustomPlugin(field_collection)


# Looks like deserilaize required? Not found in the example here: https://github.com/NVIDIA/TensorRT/blob/main/samples/python/python_plugin/circ_pad_plugin_multi_tactic.py
    # def deserialize_plugin(self, name: str, data: bytes) -> CircularPaddingPlugin:
    #     dict = pkl.loads(data)
    #     deserialized = <PLUGIN CLASS>()
    #     deserialized.__dict__.update(dict)
    #     return deserialized

TRT_PLUGIN_REGISTRY = trt.get_plugin_registry()
TRT_PLUGIN_REGISTRY.register_creator(CircularPaddingPluginCreator(), "") 