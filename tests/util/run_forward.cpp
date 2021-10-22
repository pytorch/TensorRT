#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

torch::jit::IValue RunModuleForward(torch::jit::Module& mod, std::vector<torch::jit::IValue> inputs) {
  mod.to(at::kCUDA);
  return mod.forward(inputs);
}

std::vector<at::Tensor> RunModuleForwardAsEngine(torch::jit::Module& mod, std::vector<at::Tensor> inputs) {
  auto forward_graph = mod.get_method("forward");
  std::vector<c10::ArrayRef<int64_t>> input_ranges;
  for (auto in : inputs) {
    input_ranges.push_back(in.sizes());
  }

  auto engine = torch_tensorrt::ts::convert_method_to_trt_engine(mod, "forward", input_ranges);
  return RunEngine(engine, inputs);
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
