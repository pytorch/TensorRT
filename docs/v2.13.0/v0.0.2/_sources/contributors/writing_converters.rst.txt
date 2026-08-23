.. _writing_converters:

Writing Converters
===================

Background
------------

In the JIT IR, operations are represented as nodes in a graph. A node has inputs and outputs, represented by ``torch::jit::Values``
which are typed abstract representation of data flowing into and out of a node. TensorRT represents its graph though the
use of ``nvinfer1::ILayers`` and ``nvinfer1::ITensors`` which are its analogues to nodes and values. The goal of
converters create new ILayers and subgraphs that do operation specified by the node and associate produced ITensors
and Values together.

Converters
------------

Converters should be functions which will use a list of inputs (either ``nvinfer1::ITensors`` or ``torch::jit::IValues``) to
construct an equivalent layer to the LibTorch op.

Converters can be registered using the ``RegisterNodeConversionPatterns`` helper class where you instantiate a
RegisterNodeConversionPatterns object and call the pattern function on it (like below) which takes a string
which describes the function schema of the op that will cause the converter to be run and a lambda or function
which will do the actual conversion:

    Note the pattern function can be chained

.. code-block:: c++

    auto acthardtanh TRTORCH_UNUSED = RegisterNodeConversionPatterns()
        .pattern({
            "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                auto in = args[0].ITensor();
                auto min = args[1].unwrapToDouble();
                auto max = args[2].unwrapToDouble();

                auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kCLIP);
                TRTORCH_CHECK(new_layer, "Unable to create layer for aten::hardtanh");

                new_layer->setAlpha(min);
                new_layer->setBeta(max);

                new_layer->setName(util::node_info(n).c_str());
                auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

                LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
                return true;
            }
        });


Converter Contract
----------------------

What is guaranteed to converters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. In the args there will be an entry for each node input value, either a ITensor or IValue
2. Inputs will be provided in order according to the function schema

Responsibilities of a converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.  Args must be guaranteed to be a type to unwrap the Arg union without checking, typically input tensor arguments can be expected to be ITensors
2.  Any weights or static values must guaranteed to be valid until the end of conversion time

    a. A helpful tool is the Weights helper class described below

3.  Converters are expected to produce an IValue or ITensor for each output of a node. The compiler will check this and produce warnings if there are Values that don't have associated ITensors or IValues.
4.  Outputs must be annotated

    a.  There must be an association between a JIT nodes output values and the new TRT layers output tensors in the ``value_tensor_map`` in the conversion context

5.  Name your layers

    a.  Its much easier to debug when we can track which layers and nodes correspond with each other. The system we are currently using is to use the "node info" of the node as the name of the layer

6.  Name your tensors

    a.  Use the output value debug name as the name for the new ITensor (again for debugging)

Conversion Context
--------------------

The conversion context maintains the state of conversion, it manages the Network Definition, two maps
one that stores associations between Values and IValues (the evaluated_value_map) and one that stores
associations between Values and ITensors, and any sort of memory that needs to live until the end of
conversion. The main apis that you will interface with in converters is directly accessing the network
definition to add layers ``ctx->net`` and data association functions ``ctx->AssociateValueAndTensor()``
and ``ctx->AssociateValueAndIValue()``, which you will use to add layers to the TRT layers and log
pairs of node outputs and static values or TensorRT layer outputs.

Args
-------

Arguments provided to the converter are inspectable unions of ``nvinfer1::ITensors`` and ``torch::jit::IValues`` (i.e.
abstract dataflow in the TensorRT graph and static values). You are guaranteed that you will have some
argument for each input value for the node. They are provided in the order of the function schema.
It can be expected that inputs (meaning the parameters that would be passed into the forward
function of a module in PyTorch) will be ITensors but the Arg class also has mechanisms to inspect arguments safely
before unwrapping if you are unsure. Args also have deep unwrap methods that let you get straight to the
underlying data in an IValue if you know it's safe. You can also pass in a fallback value if there is a
chance the IValue is None.

Weights
--------------

Weights are used during build time, so any weights need to be guaranteed to live until the end of the conversion phase.
TensorRT also uses its own weights structure to hold the weights. There is a wrapper around this class available
to converts which abstracts a lot of this.

The weights wrapper class can accept either ``at::Tensors`` or singular values (right now). You also need to pass the
conversion context when constructing these weights because internally the weights class will allocate memory managed
by the conversion context to store a copy of the tensor data. This data gets freed when the conversion context
destructor gets destroyed so converters don't really need to think about it.

There is metadata generated from the shape of the input data which becomes useful in interfacing with TensorRT, such
as number of input maps, number of output maps and kernel shape.

Other advice
--------------

You have the benefit of the full aten library when dealing with weights and other static values. This means that you
can do quite a bit of work during conversion time to produce efficient conversion. A good example is 2D batch_norm
converter where the converter does fusion of the batch norm operations to a conv layer in the converter using the
tensors passed in.
