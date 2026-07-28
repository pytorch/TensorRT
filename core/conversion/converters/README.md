# Converters

A Library of converters which map from LibTorch Ops to TensorRT Layers

## Writing a converter
Converters should be functions which will use a list of inputs (either `nvinfer1::ITensors` or `torch::jit::IValues`) to construct an equivalent layer to the LibTorch op.

Converters can be registered using the `RegisterNodeConversionPatterns` helper class where you instantiate a RegisterNodeConversionPatterns object and call the pattern function on it (like below) which takes a string which describes the function schema of the op and a lambda or function which will do the actual conversion:
> Note the pattern function can be chained

``` C++
bool relu(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
     auto in = args[0].ITensor();

     auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kRELU);
     if (!new_layer) {
         LOG_ERROR("Unable to create ReLU layer from node: " << *n);
         return false;
     }

     new_layer->setName(util::node_info(n).c_str());
     auto out_value = n->outputs()[0];
     auto out_tensor = new_layer->getOutput(0);
     out_tensor->setName(out_value->debugName().c_str());
     ctx->value_tensor_map[out_value] = out_tensor;
     LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

     return true;
}

auto relu_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::relu(Tensor input) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            return relu(ctx, n, args);
        }
    }).pattern({
        "aten::relu_(Tensor(a!) self) -> (Tensor(a!))",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            return relu(ctx, n, args);
        }
    });

```

### Args

Arguments provided to the converter are unions of `nvinfer1::ITensors` and `torch::jit::IValues` (i.e. abstract dataflow in the TensorRT graph and static values). You are guaranteed that you will have some argument for each input value for the node. They are provided in the order of the function schema (to be verified). It can be expected that inputs (meaning the parameters that would be passed into the forward function in PyTorch) will be ITensors but the Arg class also has mechanisms to inspect arguments safely before unwrapping if you are unsure. Args also have unwrap methods that let you get straight to the underlying data in an IValue if you know it's safe, you can also pass in a fallback value if there is a chance the IValue is None.

### Weights

Weights are used during build time, so any weights need to be guaranteed to live until the end of conversion time. TensorRT also uses its own weights structure to hold the weights. There is a wrapper around this class available to converts which abstracts a lot of this.

The weights wrapper class can accept either `at::Tensor`s or singular values (right now). You also need to pass the conversion context when constructing these weights because internally the weights class will allocate memory managed by the conversion context to store a copy of the tensor data. This data gets freed when the conversion context destructor gets destroyed so converters don't really need to think about it.

There is metadata generated from the shape of the input data which becomes useful in interfacing with TensorRT, such as number of input maps, number of output maps and kernel shape.

### Other advice

You have the benefit of the full aten library when dealing with weights and other static values. This means that you can do quite a bit of work during conversion time to produce efficient conversion. A good example is 2D batch_norm converter where the converter does fusion of the batch norm operations to a conv layer in the converter using the tensors passed in.

## Converter Contract

Here is what is guaranteed to converters

1. In the args there will be an entry for each node input value, either a ITensor or IValue
2. **Need to verify for sure** Inputs will always be provided in order according to the function schema

Here are the responsibilities of a converter

1.  Args must be guaranteed to be a type to unwrap the Arg union without checking, typically input arguments can be expected to be ITensors
2. Any weights or static values must guaranteed to be valid until the end of conversion time
   1. A helpful tool is the Weights helper class described above
3. Outputs must be annotated
   1. There must be an association between a JIT nodes output values and the new TRT layers output tensors in the value_tensor_map in the conversion context
4. Name your layers
   1. Its much easier to debug when we can track which layers and nodes correspond with each other.  The system we are currently using is to use the node_info of the node as the name of the layer
5. Name your tensors
   1. Use the output value debug name as the name for the new ITensor (again for debugging)
