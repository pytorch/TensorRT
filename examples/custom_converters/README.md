# Create a new op in C++, compile it to .so library and load it in Python

There are some operators in PyTorch library which are not supported in TRTorch. 
To support these ops, users can register converters for missing ops. For example,
if we try to compile a graph with a build of TRTorch that doesn't support the 
[ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) operation, 
we will get following error:

> Unable to convert node: %result.2 : Tensor = aten::elu(%x.1, %2, %3, %3) # /home/bowa/.local/lib/python3.6/site-packages/torch/nn/functional.py:1227:17 (conversion.AddLayer)
Schema: aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)
Converter for aten::elu requested, but no such converter was found.
If you need a converter for this operator, you can try implementing one yourself
or request a converter: https://www.github.com/NVIDIA/TRTorch/issues

## Writing Converter in C++
We can register a converter for this operator in our application. You can find more 
information on all the details of writing converters in the contributors documentation
([Writing Converters](https://nvidia.github.io/TRTorch/contributors/writing_converters.html)).
Once we are clear about these rules and writing patterns, we can create a seperate new C++ source file as:

```c++
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace my_custom_converters {

auto actelu = trtorch::core::conversion::converters::RegisterNodeConversionPatterns().pattern(
    {"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)",
     [](trtorch::core::conversion::ConversionCtx* ctx,
        const torch::jit::Node* n,
        trtorch::core::conversion::converters::args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       auto alpha = args[1].unwrapToDouble();

       auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kELU);
       if (!(new_layer)) {
         std::cerr << "Unable to create layer for aten::elu" << std::endl;
       }

       new_layer->setAlpha(alpha);
       new_layer->setName(trtorch::core::util::node_info(n).c_str());
       ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

       return true;
     }});

} // namespace my_custom_converters
```

## Generate `.so` library
To use this converter in Python, it is recommended to use PyTorch's
[C++/CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions).
We give an example here about how to wrap the converter into a `.so`
library so that you can load it to use in Python applicaton. 
```python
import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

dir_path = os.path.dirname(os.path.realpath(__file__))

# library_dirs should point to the libtrtorch.so, include_dirs should point to the dir that include the headers
# 1) download the latest package from https://github.com/NVIDIA/TRTorch/releases/
# 2) Extract the file from downloaded package, we will get the "trtorch" directory
# 3) Set trtorch_path to that directory
trtorch_path = os.path.abspath("trtorch")

ext_modules = [
    cpp_extension.CUDAExtension('elu_converter', ['elu_converter.cpp'],
                                library_dirs=[(trtorch_path + "/lib/")],
                                libraries=["trtorch"],
                                include_dirs=[trtorch_path + "/include/trtorch/"])
]

setup(
    name='elu_converter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
```
Make sure to include the path for header files in `include_dirs` and the path 
for dependent libraries in `library_dirs`. Generally speaking, you should download 
the latest package from [here](https://github.com/NVIDIA/TRTorch/releases), extract
the files, and the set the `trtorch_path` to it. You could also add other compilation 
flags in cpp_extension if you need. Then, run above python scripts as:
```shell
python3 setup.py install --user
```
You should see the output similar to the contents indicated [here](https://pytorch.org/tutorials/advanced/cpp_extension.html#custom-c-and-cuda-extensions)   after running
`python setup.py install`. You should find a couple of new folders generated
by the command above. In build folder, you can find the generated `.so` library, 
which could be loaded in our Python application. 

## Load `.so` in Python Application
With the new generated library, TRTorch now support the new developed converter. 
We use `torch.ops.load_library` to load `.so`. For example, we could load the ELU 
converter and use it in our application:
```python
import torch
import trtorch

# After "python3 setup install", you should find this .so file under generated "build" directory
torch.ops.load_library('./build/lib.linux-x86_64-3.6/elu_converter.cpython-36m-x86_64-linux-gnu.so')


class Elu(torch.nn.Module):

    def __init__(self):
        super(Elu, self).__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)


def MaxDiff(pytorch_out, trtorch_out):
    diff = torch.sub(pytorch_out, trtorch_out)
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff)
    print("Maximum differnce between TRTorch and PyTorch: \n", max_diff)


def main():
    model = Elu().eval()  #.cuda()

    scripted_model = torch.jit.script(model)
    compile_settings = {
        "input_shapes": [{
            "min": [1024, 1, 32, 32],
            "opt": [1024, 1, 33, 33],
            "max": [1024, 1, 34, 34],
        }],
        "op_precision":
            torch.half  # Run with FP16
    }
    trt_ts_module = trtorch.compile(scripted_model, compile_settings)
    input_data = torch.randn((1024, 1, 32, 32))
    input_data = input_data.half().to("cuda")
    pytorch_out = model.forward(input_data)

    trtorch_out = trt_ts_module(input_data)
    print('PyTorch output: \n', pytorch_out[0, :, :, 0])
    print('TRTorch output: \n', trtorch_out[0, :, :, 0])
    MaxDiff(pytorch_out, trtorch_out)


if __name__ == "__main__":
    main()

```
Run this script, we can get the different outputs from PyTorch and TRTorch. 
