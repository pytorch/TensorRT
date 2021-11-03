# Torch-TensorRT Core
The Torch-TensorRT Core is the main graph analysis library, it processes a TorchScript Module, converting method graphs to engines and returning a new equivalent module which when run will run inputs through a TensorRT engine

## Stages

> Basic rule of thumb for organization, if the the output of the component is a modified block then it is in lowering, if the output is a TRT engine block then its in conversion

## Lowering Passes

There are a set of passes over the IR that will be made to lower the graph into a block of convertible nodes.

### PyTorch JIT Lowering Passes
Firstly the graph will go through the lowering passes used in LibTorch, this will lower it to a graph where all attributes accesses are replaced with explicit inputs to the graph (i.e. graph parameters vs. prim::GetAttr)

#### Call Method Insertions

Graphs from prim::CallMethods need to be inserted into the graph or used to segment the graph into convertible subgraphs.

### Torch-TensorRT Lowering

To simplify conversion we can use the PyTorch JIT Subgraph Rewriter to simplify the set of subgraphs that need explicit TensorRT converters. This means we could aim for closer to 1->1 op conversion vs looking for applicable subgraphs, limit the number of converters and reduce the size of each converter.


## Conversion Phase

Once the graph has be simplified to a form thats easy to convert, we then set up a conversion context to manage the construction of a TensorRT INetworkDefinition from the blocks nodes. The conversion context records the set of converted nodes, block inputs and outputs and other information about the conversion of the graph. This data is then used to help converters link together layers and also hold build time information like weights required to construct the engine. After the context is created, the block converter starts iterating through the list of nodes, for each node, the converter will look at its inputs and assemble a dictionary of resources to pass to the converter. Inputs can be in a couple of states:
- The input is a block parameter
  In this case the input should have already been stored in as an IValue in the conversion context evaluated_value_map. The conversion stage will add the IValue to the list of args for the converter
- The input is an output of a node that has already been converted
  In this case the ITensor of the output has added to the to the value_tensor_map, The conversion stage will add the ITensor to the list of args for the converter
- The input is from a node that produces a static value
  There are nodes that produce static values, typically used to store parameters for operators, we need to evaluate these nodes at conversion time to be able to convert a op. The converter will look for a node evaluator in the evaluator registry and run it on the node. The IValue produced will be entered in the conversion context evaluated_value_map and added to the list of args for the converter.
- The input is from a node that has not been converted
  Torch-TensorRT will error here

### Node Evaluation

There are some nodes that contain static data and are resources for operations. These can be evaluated at conversion time so that you can use those values when doing node conversion. In theory any node kind can have a conversion time evaluator as long as it produces a static IValue, This IValue will be stored in the conversion context so it can be consumed by any node that takes the evaluated node as an input.

### Converters

See the README in //core/conversion/converters for more information
