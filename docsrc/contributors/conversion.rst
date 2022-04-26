.. _conversion:

Conversion Phase
==================

Once the graph has be simplified to a form thats easy to convert, we then set up a conversion context
to manage the construction of a TensorRT ``INetworkDefinition`` from the blocks nodes. The conversion context
records the set of converted nodes, block inputs and outputs and other information about the conversion
of the graph. This data is then used to help converters link together layers and also hold build time
information like weights required to construct the engine. After the context is created, the block
converter starts iterating through the list of nodes, for each node, the converter will look at its
inputs and assemble an array of resources to pass to the converter. Inputs can be in a couple of states:

*  The input is a block parameter

   *  In this case the input should have already been stored in as an IValue in the
      conversion context ``evaluated_value_map``. The conversion stage will add the IValue to the list of args for the
      converter

*  The input is an output of a node that has already been converted

   *  In this case the ITensor of the output has added to the ``value_tensor_map``,
      The conversion stage will add the ITensor to the list of args for the converter

*  The input is from a node that produces a static value

   *  There are nodes that produce static values, typically used to store parameters for operators, we need to
      evaluate these nodes at conversion time to be able to convert a op. The conversion system will look for a node
      evaluator in the evaluator registry and run it on the node. The IValue produced will be entered in the
      conversion context ``evaluated_value_map`` and added to the list of args for the converter. If the node
      to be evaluated takes inputs, the conversion stage will recursively resolve dependencies until the final
      static value has been evaluated

*  The input is from a node that has not been converted

   *  Torch-TensorRT will error out here

Node Evaluation
-----------------
There are some nodes that contain static data and are resources for operations. These can be evaluated at
conversion time so that you can use those values when doing node conversion. In theory any node kind can have
a conversion time evaluator as long as it produces a static IValue, This IValue will be stored in the conversion
context so it can be consumed by any node that takes the evaluated node as an input. Common node types are
``prim::Constant`` which emits a constant and ``prim::ListConstruct`` which makes lists.

Node Converters
----------------

Node converters map JIT nodes to layers or subgraphs of layers. They then associate outputs from the JIT graph
and the TRT graph together in the conversion context. This allows the conversion stage to assemble the inputs
for the next node. There are some cases where a node produces an output that is not a Tensor but a static result
from a calculation done on inputs which need to be converted first. In this case the converter may associate the outputs in
the ``evaluated_value_map`` instead of the ``value_tensor_map``. For more information take a look at: :ref:`writing_converters`
