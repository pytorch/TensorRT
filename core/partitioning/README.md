# Torch-TensorRT Partitioning

Torch-TensorRT partitioning phase is developed to support `automatic fallback` feature in Torch-TensorRT. This phase won't run by
default until the automatic fallback feature is enabled.

On a high level, Torch-TensorRT partitioning phase does the following:
- `Segmentation`. Go through the set of operators in order and verify if there is converter for each operator. Then,
roughly separate the graph into parts that Torch-TensorRT can support and parts Torch-TensorRT cannot.
- `Dependency Analysis`. For every to be compiled operator there is a "complete dependency graph", which means that
every input can to traced back to an input as Tensor or TensorList. Go through all segments after segmentation then
  do dependency analysis to ensure that there are only Tensor/TensorList inputs and outputs for TensorRT segments.
- `Shape Analysis`. For each segments, figure out the input and outputs shapes starting from the provided input shape
from the user. Shapes can be calculated by running the graphs with JIT.
- `Conversion`. Every TensorRT segments will be converted to TensorRT engine. This part is done in compiler.cpp, but
  it's still a phase in our partitioning process.
- `Stitching`. Stitch all TensorRT engines with PyTorch nodes altogether.

Test cases for each of these components could be found [here](https://github.com/pytorch/TensorRT/tree/master/tests/core/partitioning).

Here is the brief description of functionalities of each file:
- `PartitionInfo.h/cpp`: The automatic fallback APIs that is used for partitioning.
- `SegmentedBlock.h/cpp`: The main data structures that is used to maintain information for each segments after segmentation.
- `shape_analysis.h/cpp`: Code implementation to get the shapes for each segments by running them in JIT.
- `partitioning.h/cpp`: APIs and main code implementation for partitioning phase.

### Automatic Fallback
To enable automatic fallback feature, you can set following attributes in Python:
```python
  import torch
  import torch_tensorrt as torchtrt

  ...
  model = MyModel()
  ts_model = torch.jit.script(model)
  trt_model = torchtrt.ts.compile(model, **{
    ...
    "min_block_size" : 3,
    "torch_executed_ops": ["aten::add"],
    "torch_executed_modules": [],
  })
```
- `enabled`: By default automatic fallback will be off. It is enabled by setting it to True.
- `min_block_size`: The minimum number of consecutive operations that must satisfy to be converted to TensorRT. For
example, if it's set to 3, then there must be 3 consecutive supported operators then this segments will be converted.
- `forced_fallback_ops`: A list of strings that will be the names of operations that the user explicitly want to be in
PyTorch nodes.

To enable automatic fallback feature in C++, following APIs could be uses:

```c++
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

...
auto in = torch::randn({1, 3, 224, 224}, {torch::kCUDA});

auto mod = torch::jit::load("trt_ts_module.ts");
auto input_sizes =  std::vector<torchtrt::InputRange>{{in.sizes()}};
torchtrt::ts::CompileSpec cfg(input_sizes);
cfg.min_block_size = 2;
cfg.torch_executed_ops.push_back("aten::relu");
auto trt_mod = torchtrt::ts::compile(mod, cfg);
auto out = trt_mod.forward({in});
```
