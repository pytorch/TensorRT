.. _getting_started:

Getting Started with C++
========================

If you haven't already, acquire a tarball of the library by following the instructions in :ref:`Installation`

Using Torch-TensorRT in C++
***************************
Torch-TensorRT C++ API accepts TorchScript modules (generated either from ``torch.jit.script`` or ``torch.jit.trace``) as an input and returns
a Torchscript module (optimized using TensorRT). This requires users to use Pytorch (in python) to generate torchscript modules beforehand.
Please refer to `Creating TorchScript modules in Python <https://nvidia.github.io/Torch-TensorRT/tutorials/creating_torchscript_module_in_python.html>`_ section to generate torchscript graphs.


.. _torch_tensorrt_quickstart:

[Torch-TensorRT Quickstart] Compiling TorchScript Modules with ``torchtrtc``
---------------------------------------------------------------------

An easy way to get started with Torch-TensorRT and to check if your model can be supported without extra work is to run it through
``torchtrtc``, which supports almost all features of the compiler from the command line including post training quantization
(given a previously created calibration cache). For example we can compile our lenet model by setting our preferred operating
precision and input size. This new TorchScript file can be loaded into Python (note: you need to ``import torch_tensorrt`` before loading
these compiled modules because the compiler extends the PyTorch the deserializer and runtime to execute compiled modules).

.. code-block:: shell

    ❯ torchtrtc -p f16 lenet_scripted.ts trt_lenet_scripted.ts "(1,1,32,32)"

    ❯ python3
    Python 3.6.9 (default, Apr 18 2020, 01:56:04)
    [GCC 8.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import torch
    >>> import torch_tensorrt
    >>> ts_model = torch.jit.load(“trt_lenet_scripted.ts”)
    >>> ts_model(torch.randn((1,1,32,32)).to(“cuda”).half())

You can learn more about ``torchtrtc`` usage here: :ref:`torchtrtc`

.. _ts_in_cc:

Working with TorchScript in C++
--------------------------------

If we are developing an application to deploy with C++, we can save either our traced or scripted module using ``torch.jit.save``
which will serialize the TorchScript code, weights and other information into a package. This is also where our dependency on Python ends.

.. code-block:: python

    torch_script_module.save("lenet.jit.pt")

From here we can now load our TorchScript module in C++

.. code-block:: c++

    #include <torch/script.h> // One-stop header.

    #include <iostream>
    #include <memory>

    int main(int argc, const char* argv[]) {
        torch::jit::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load("<PATH TO SAVED TS MOD>");
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }

        std::cout << "ok\n";


You can do full training and inference in C++ with PyTorch / LibTorch if you would like, you can even define your modules in C++ and
have access to the same powerful tensor library that backs PyTorch. (For more information: https://pytorch.org/cppdocs/).
For instance we can do inference with our LeNet module like this:

.. code-block:: c++

        mod.eval();
        torch::Tensor in = torch::randn({1, 1, 32, 32});
        auto out = mod.forward(in);

and to run on the GPU:

.. code-block:: c++

        mod.eval();
        mod.to(torch::kCUDA);
        torch::Tensor in = torch::randn({1, 1, 32, 32}, torch::kCUDA);
        auto out = mod.forward(in);

As you can see it is pretty similar to the Python API. When you call the ``forward`` method, you invoke the PyTorch JIT compiler, which will optimize and run your TorchScript code.

.. _compile_cpp:

Compiling with Torch-TensorRT in C++
-------------------------------------
We are also at the point were we can compile and optimize our module with Torch-TensorRT, but instead of in a JIT fashion we must do it ahead-of-time (AOT) i.e. before we start doing actual inference work
since it takes a bit of time to optimize the module, it would not make sense to do this every time you run the module or even the first time you run it.

With our module loaded, we can feed it into the Torch-TensorRT compiler. When we do so we must provide some information on the expected input size and also configure any additional settings.

.. code-block:: c++

    #include "torch/script.h"
    #include "torch_tensorrt/torch_tensorrt.h"
    ...

        mod.to(at::kCUDA);
        mod.eval();

        auto in = torch::randn({1, 1, 32, 32}, {torch::kCUDA});
        auto trt_mod = torch_tensorrt::CompileGraph(mod, std::vector<torch_tensorrt::CompileSpec::InputRange>{{in.sizes()}});
        auto out = trt_mod.forward({in});

Thats it! Now the graph runs primarily not with the JIT compiler but using TensorRT (though we execute the graph using the JIT runtime).

We can also set settings like operating precision to run in FP16.

.. code-block:: c++

    #include "torch/script.h"
    #include "torch_tensorrt/torch_tensorrt.h"
    ...

        mod.to(at::kCUDA);
        mod.eval();

        auto in = torch::randn({1, 1, 32, 32}, {torch::kCUDA}).to(torch::kHALF);
        auto input_sizes = std::vector<torch_tensorrt::CompileSpec::InputRange>({in.sizes()});
        torch_tensorrt::CompileSpec info(input_sizes);
        info.enable_precisions.insert(torch::kHALF);
        auto trt_mod = torch_tensorrt::CompileGraph(mod, info);
        auto out = trt_mod.forward({in});

And now we are running the module in FP16 precision. You can then save the module to load later.

.. code-block:: c++

    trt_mod.save("<PATH TO SAVED TRT/TS MOD>")

Torch-TensorRT compiled TorchScript modules are loaded in the same way as normal TorchScript module. Make sure your deployment application is linked against ``libtorchtrt.so``

.. code-block:: c++

    #include "torch/script.h"
    #include "torch_tensorrt/torch_tensorrt.h"

    int main(int argc, const char* argv[]) {
        torch::jit::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load("<PATH TO SAVED TRT/TS MOD>");
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }

        torch::Tensor in = torch::randn({1, 1, 32, 32}, torch::kCUDA);
        auto out = mod.forward(in);

        std::cout << "ok\n";
    }

If you want to save the engine produced by Torch-TensorRT to use in a TensorRT application you can use the ``ConvertGraphToTRTEngine`` API.

.. code-block:: c++

    #include "torch/script.h"
    #include "torch_tensorrt/torch_tensorrt.h"
    ...

        mod.to(at::kCUDA);
        mod.eval();

        auto in = torch::randn({1, 1, 32, 32}, {torch::kCUDA}).to(torch::kHALF);
        auto input_sizes = std::vector<torch_tensorrt::CompileSpec::InputRange>({in.sizes()});
        torch_tensorrt::CompileSpec info(input_sizes);
        info.enabled_precisions.insert(torch::kHALF);
        auto trt_mod = torch_tensorrt::ConvertGraphToTRTEngine(mod, "forward", info);
        std::ofstream out("/tmp/engine_converted_from_jit.trt");
        out << engine;
        out.close();

.. _under_the_hood:

Under The Hood
---------------

When a module is provided to Torch-TensorRT, the compiler starts by mapping a graph like you saw above to a graph like this:

.. code-block:: none

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

The graph has now been transformed from a collection of modules, each managing their own parameters into a single graph with the parameters inlined
into the graph and all of the operations laid out. Torch-TensorRT has also executed a number of optimizations and mappings to make the graph easier to translate to TensorRT.
From here the compiler can assemble the TensorRT engine by following the dataflow through the graph.

When the graph construction phase is complete, Torch-TensorRT produces a serialized TensorRT engine. From here depending on the API, this engine is returned
to the user or moves into the graph construction phase. Here Torch-TensorRT creates a JIT Module to execute the TensorRT engine which will be instantiated and managed
by the Torch-TensorRT runtime.

Here is the graph that you get back after compilation is complete:

.. code-block:: none

    graph(%self_1 : __torch__.lenet, %input_0 : Tensor):
        %1 : ...trt.Engine = prim::GetAttr[name="lenet"](%self_1)
        %3 : Tensor[] = prim::ListConstruct(%input_0)
        %4 : Tensor[] = trt::execute_engine(%3, %1)
        %5 : Tensor = prim::ListUnpack(%4)
        return (%5)


You can see the call where the engine is executed, after extracting the attribute containing the engine and constructing a list of inputs, then returns the tensors back to the user.

.. _unsupported_ops:

Working with Unsupported Operators
-----------------------------------

Torch-TensorRT is a new library and the PyTorch operator library is quite large, so there will be ops that aren't supported natively by the compiler. You can either use the composition techinques
shown above to make modules are fully Torch-TensorRT supported and ones that are not and stitch the modules together in the deployment application or you can register converters for missing ops.

    You can check support without going through the full compilation pipleine using the ``torch_tensorrt::CheckMethodOperatorSupport(const torch::jit::Module& module, std::string method_name)`` api
    to see what operators are not supported. ``torchtrtc`` automatically checks modules with this method before starting compilation and will print out a list of operators that are not supported.

.. _custom_converters:

Registering Custom Converters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operations are mapped to TensorRT through the use of modular converters, a function that takes a node from a the JIT graph and produces an equivalent layer or subgraph in TensorRT.
Torch-TensorRT ships with a library of these converters stored in a registry, that will be executed depending on the node being parsed. For instance a ``aten::relu(%input0.4)`` instruction will trigger
the relu converter to be run on it, producing an activation layer in the TensorRT graph. But since this library is not exhaustive you may need to write your own to get Torch-TensorRT
to support your module.

Shipped with the Torch-TensorRT distribution are the internal core API headers. You can therefore access the converter registry and add a converter for the op you need.

For example, if we try to compile a graph with a build of Torch-TensorRT that doesn't support the flatten operation (``aten::flatten``) you may see this error:

.. code-block:: none

    terminate called after throwing an instance of 'torch_tensorrt::Error'
    what():  [enforce fail at core/conversion/conversion.cpp:109] Expected converter to be true but got false
    Unable to convert node: %input.1 : Tensor = aten::flatten(%x.1, %11, %5) # x.py:25:0 (conversion.AddLayer)
    Schema: aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)
    Converter for aten::flatten requested, but no such converter was found.
    If you need a converter for this operator, you can try implementing one yourself
    or request a converter: https://www.github.com/NVIDIA/Torch-TensorRT/issues

We can register a converter for this operator in our application. All of the tools required to build a converter can be imported by including ``torch_tensorrt/core/conversion/converters/converters.h``.
We start by creating an instance of the self-registering class ``torch_tensorrt::core::conversion::converters::RegisterNodeConversionPatterns()`` which will register converters
in the global converter registry, associating a function schema like ``aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)`` with a lambda that
will take the state of the conversion, the node/operation in question to convert and all of the inputs to the node and produces as a side effect a new layer in the TensorRT network.
Arguments are passed as a vector of inspectable unions of TensorRT ``ITensors`` and Torch ``IValues`` in the order arguments are listed in the schema.

Below is a implementation of a ``aten::flatten`` converter that we can use in our application. You have full access to the Torch and TensorRT libraries in the converter implementation. So
for example we can quickly get the output size by just running the operation in PyTorch instead of implementing the full calculation outself like we do below for this flatten converter.

.. code-block:: c++

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

    int main() {
        ...

To use this converter in Python, it is recommended to use PyTorch's `C++ / CUDA Extention <https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions>`_
template to wrap your library of converters into a ``.so`` that you can load with ``ctypes.CDLL()`` in your Python application.

You can find more information on all the details of writing converters in the contributors documentation (:ref:`writing_converters`).
If you find yourself with a large library of converter implementations, do consider upstreaming them, PRs are welcome and it would be great for the community to benefit as well.
