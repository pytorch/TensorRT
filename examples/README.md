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

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto acthardtanh TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       auto alpha = args[1].unwrapToDouble();

       auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kELU);
       TRTORCH_CHECK(new_layer, "Unable to create layer for aten::elu");

       new_layer->setAlpha(alpha);
       new_layer->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

       LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
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

torch.ops.load_library('./build/lib.linux-x86_64-3.6/elu_converter.cpython-36m-x86_64-linux-gnu.so')

class Elu(torch.nn.Module):
    def __init__(self):
        super(Elu, self).__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)

def main():
    data = torch.randn((1, 1, 2, 2)).to("cuda")
    model = Elu().eval()  #.cuda()

    scripted_model = torch.jit.script(model)
    print(scripted_model.graph)
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
    print(input_data[0, :, :, 0])
    input_data = input_data.half().to("cuda")
    result = trt_ts_module(input_data)
    print(result[0, :, :, 0])

if __name__ == "__main__":
    main()

```
Run this script, we can get the Tensor before and after ELU operator.
### Example Output
```bash
PyTorch output: 
 tensor([[ 0.8804,  2.4355, -0.7920, -0.2070, -0.5352,  0.4775,  1.3604, -0.3350,
         -0.1802, -0.7563, -0.1758,  0.4067,  1.2510, -0.7100, -0.6221, -0.7207,
         -0.1118,  0.9966,  1.6396, -0.1367, -0.5742,  0.5859,  0.8511,  0.6572,
         -0.3481,  0.5933, -0.0488, -0.4287, -0.4102, -0.7402,  0.7515, -0.7710]],
       device='cuda:0', dtype=torch.float16)
TRTorch output: 
 tensor([[ 0.8804,  2.4355, -0.7920, -0.2070, -0.5356,  0.4775,  1.3604, -0.3347,
         -0.1802, -0.7563, -0.1758,  0.4067,  1.2510, -0.7100, -0.6221, -0.7207,
         -0.1117,  0.9966,  1.6396, -0.1368, -0.5747,  0.5859,  0.8511,  0.6572,
         -0.3484,  0.5933, -0.0486, -0.4285, -0.4102, -0.7402,  0.7515, -0.7710]],
       device='cuda:0', dtype=torch.float16)
Maximum differnce between TRTorch and PyTorch: 
 tensor(0.0005, device='cuda:0', dtype=torch.float16)


```
