# torch_tensorrt

> Ahead of Time (AOT) compiling for PyTorch JIT

Torch-TensorRT is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, Torch-TensorRT is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. Torch-TensorRT operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module.

## Example Usage

``` python
import torch_tensorrt

...

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [example_tensor, # Provide example tensor for input shape or...
        torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 224, 224],
            opt_shape=[1, 3, 512, 512],
            max_shape=[1, 3, 1024, 1024],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16)

result = trt_ts_module(input_data) # run inference
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript

```

## Installation

| ABI / Platform                          | Installation command                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| Pre CXX11 ABI (Linux x86_64)            | python3 setup.py install                                     |
| CXX ABI  (Linux x86_64)                 | python3 setup.py install --use-cxx11-abi                     |
| Pre CXX11 ABI (Jetson platform aarch64) | python3 setup.py install --jetpack-version 4.6               |
| CXX11 ABI (Jetson platform aarch64)     | python3 setup.py install --jetpack-version 4.6 --use-cxx11-abi |

For Linux x86_64 platform, Pytorch libraries default to pre cxx11 abi. So, please use `python3 setup.py install`.

On Jetson platforms, NVIDIA hosts <a href="https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048">pre-built Pytorch wheel files</a>. These wheel files are built with CXX11 ABI. So on jetson platforms, please use `python3 setup.py install --jetpack-version 4.6 --use-cxx11-abi`

## Under the Hood

When a traced module is provided to Torch-TensorRT, the compiler takes the internal representation and transforms it into one like this:

```
graph(%input.2 : Tensor):
    %2 : Float(84, 10) = prim::Constant[value=<Tensor>]()
    %3 : Float(120, 84) = prim::Constant[value=<Tensor>]()
    %4 : Float(576, 120) = prim::Constant[value=<Tensor>]()
    %5 : int = prim::Constant[value=-1]() # x.py:25:0
    %6 : int[] = prim::Constant[value=annotate(List[int], [])]()
    %7 : int[] = prim::Constant[value=[2, 2]]()
    %8 : int[] = prim::Constant[value=[0, 0]]()
    %9 : int[] = prim::Constant[value=[1, 1]]()
    %10 : bool = prim::Constant[value=1]() # ~/.local/lib/python3.6/site-packages/torch/nn/modules/conv.py:346:0
    %11 : int = prim::Constant[value=1]() # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:539:0
    %12 : bool = prim::Constant[value=0]() # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:539:0
    %self.classifer.fc3.bias : Float(10) = prim::Constant[value= 0.0464  0.0383  0.0678  0.0932  0.1045 -0.0805 -0.0435 -0.0818  0.0208 -0.0358 [ CUDAFloatType{10} ]]()
    %self.classifer.fc2.bias : Float(84) = prim::Constant[value=<Tensor>]()
    %self.classifer.fc1.bias : Float(120) = prim::Constant[value=<Tensor>]()
    %self.feat.conv2.weight : Float(16, 6, 3, 3) = prim::Constant[value=<Tensor>]()
    %self.feat.conv2.bias : Float(16) = prim::Constant[value=<Tensor>]()
    %self.feat.conv1.weight : Float(6, 1, 3, 3) = prim::Constant[value=<Tensor>]()
    %self.feat.conv1.bias : Float(6) = prim::Constant[value= 0.0530 -0.1691  0.2802  0.1502  0.1056 -0.1549 [ CUDAFloatType{6} ]]()
    %input0.4 : Tensor = aten::_convolution(%input.2, %self.feat.conv1.weight, %self.feat.conv1.bias, %9, %8, %9, %12, %8, %11, %12, %12, %10) # ~/.local/lib/python3.6/site-packages/torch/nn/modules/conv.py:346:0
    %input0.5 : Tensor = aten::relu(%input0.4) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:1063:0
    %input1.2 : Tensor = aten::max_pool2d(%input0.5, %7, %6, %8, %9, %12) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:539:0
    %input0.6 : Tensor = aten::_convolution(%input1.2, %self.feat.conv2.weight, %self.feat.conv2.bias, %9, %8, %9, %12, %8, %11, %12, %12, %10) # ~/.local/lib/python3.6/site-packages/torch/nn/modules/conv.py:346:0
    %input2.1 : Tensor = aten::relu(%input0.6) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:1063:0
    %x.1 : Tensor = aten::max_pool2d(%input2.1, %7, %6, %8, %9, %12) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:539:0
    %input.1 : Tensor = aten::flatten(%x.1, %11, %5) # x.py:25:0
    %27 : Tensor = aten::matmul(%input.1, %4)
    %28 : Tensor = trt::const(%self.classifer.fc1.bias)
    %29 : Tensor = aten::add_(%28, %27, %11)
    %input0.2 : Tensor = aten::relu(%29) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:1063:0
    %31 : Tensor = aten::matmul(%input0.2, %3)
    %32 : Tensor = trt::const(%self.classifer.fc2.bias)
    %33 : Tensor = aten::add_(%32, %31, %11)
    %input1.1 : Tensor = aten::relu(%33) # ~/.local/lib/python3.6/site-packages/torch/nn/functional.py:1063:0
    %35 : Tensor = aten::matmul(%input1.1, %2)
    %36 : Tensor = trt::const(%self.classifer.fc3.bias)
    %37 : Tensor = aten::add_(%36, %35, %11)
    return (%37)
(CompileGraph)
```

The graph has now been transformed from a collection of modules much like how your PyTorch Modules are collections of modules, each managing their own parameters into a single graph
with the parameters inlined into the graph and all of the operations laid out. Torch-TensorRT has also executed a number of optimizations and mappings to make the graph easier to translate
to TensorRT. From here the compiler can assemble the TensorRT engine by following the dataflow through the graph.

When the graph construction phase is complete, Torch-TensorRT produces a serialized TensorRT engine. From here depending on the API, this engine is returned to the user or moves into the graph
construction phase. Here Torch-TensorRT creates a JIT Module to execute the TensorRT engine which will be instantiated and managed by the Torch-TensorRT runtime.

Here is the graph that you get back after compilation is complete:

```
graph(%self.1 : __torch__.___torch_mangle_10.LeNet_trt,
    %2 : Tensor):
    %1 : int = prim::Constant[value=94106001690080]()
    %3 : Tensor = trt::execute_engine(%1, %2)
    return (%3)
(AddEngineToGraph)
```

You can see the call where the engine is executed, based on a constant which is the ID of the engine, telling JIT how to find the engine and the input tensor which will be fed to TensorRT.
The engine represents the exact same calculations as what is done by running a normal PyTorch module but optimized to run on your GPU.

Torch-TensorRT converts from TorchScript by generating layers or subgraphs in correspondance with instructions seen in the graph. Converters are small modules of code used to map one specific
operation to a layer or subgraph in TensorRT. Not all operations are support, but if you need to implement one, you can in C++.

## Registering Custom Converters

Operations are mapped to TensorRT through the use of modular converters, a function that takes a node from a the JIT graph and produces an equivalent layer or subgraph in TensorRT. Torch-TensorRT
ships with a library of these converters stored in a registry, that will be executed depending on the node being parsed. For instance a `aten::relu(%input0.4)` instruction will trigger the
relu converter to be run on it, producing an activation layer in the TensorRT graph. But since this library is not exhaustive you may need to write your own to get Torch-TensorRT to support your module.

Shipped with the Torch-TensorRT distribution are the internal core API headers. You can therefore access the converter registry and add a converter for the op you need.

For example, if we try to compile a graph with a build of Torch-TensorRT that doesn’t support the flatten operation (`aten::flatten`) you may see this error:

```
terminate called after throwing an instance of 'torch_tensorrt::Error'
what():  [enforce fail at core/conversion/conversion.cpp:109] Expected converter to be true but got false
Unable to convert node: %input.1 : Tensor = aten::flatten(%x.1, %11, %5) # x.py:25:0 (conversion.AddLayer)
Schema: aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)
Converter for aten::flatten requested, but no such converter was found.
If you need a converter for this operator, you can try implementing one yourself
or request a converter: https://www.github.com/NVIDIA/Torch-TensorRT/issues
```

We can register a converter for this operator in our application. All of the tools required to build a converter can be imported by including `Torch-TensorRT/core/conversion/converters/converters.h`.
We start by creating an instance of the self-registering `class torch_tensorrt::core::conversion::converters::RegisterNodeConversionPatterns()` which will register converters in the global converter
registry, associating a function schema like `aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)` with a lambda that will take the state of the conversion, the
node/operation in question to convert and all of the inputs to the node and produces as a side effect a new layer in the TensorRT network. Arguments are passed as a vector of inspectable unions
of TensorRT ITensors and Torch IValues in the order arguments are listed in the schema.

Below is a implementation of a `aten::flatten` converter that we can use in our application. You have full access to the Torch and TensorRT libraries in the converter implementation. So for example
we can quickly get the output size by just running the operation in PyTorch instead of implementing the full calculation outself like we do below for this flatten converter.

```c++
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"
#include "torch_tensorrt/core/conversion/converters/converters.h"

static auto flatten_converter = torch_tensorrt::core::conversion::converters::RegisterNodeConversionPatterns()
    .pattern({
        "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)",
        [](torch_tensorrt::core::conversion::ConversionCtx* ctx,
           const torch::jit::Node* n,
           torch_tensorrt::core::conversion::converters::args& args) -> bool {
            auto in = args[0].ITensor();
            auto start_dim = args[1].unwrapToInt();
            auto end_dim = args[2].unwrapToInt();
            auto in_shape = torch_tensorrt::core::util::toVec(in->getDimensions());
            auto out_shape = torch::flatten(torch::rand(in_shape), start_dim, end_dim).sizes();

            auto shuffle = ctx->net->addShuffle(*in);
            shuffle->setReshapeDimensions(torch_tensorrt::core::util::toDims(out_shape));
            shuffle->setName(torch_tensorrt::core::util::node_info(n).c_str());

            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
            return true;
        }
    });
```

To use this converter in Python, it is recommended to use PyTorch’s [C++ / CUDA Extention](https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions) template to wrap
your library of converters into a `.so` that you can load with `ctypes.CDLL()` in your Python application.

You can find more information on all the details of writing converters in the contributors documentation ([Writing Converters](https://nvidia.github.io/Torch-TensorRT/contributors/writing_converters.html#writing-converters)). If you
find yourself with a large library of converter implementations, do consider upstreaming them, PRs are welcome and it would be great for the community to benefit as well.
