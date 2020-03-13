#include "torch/script.h"
#include "trtorch/trtorch.h"
#include "tests/util/util.h"

namespace trtorch {
namespace tests {
namespace util {

torch::jit::IValue RunModuleForward(torch::jit::script::Module& mod, std::vector<torch::jit::IValue> inputs) {
    mod.to(at::kCUDA);
    return mod.forward(inputs);
}


std::vector<at::Tensor> RunModuleForwardAsEngine(torch::jit::script::Module& mod, std::vector<at::Tensor> inputs) {
    auto forward_graph = mod.get_method("forward");
    std::vector<c10::ArrayRef<int64_t>> input_ranges;
    for (auto in : inputs) {
        input_ranges.push_back(in.sizes());
    }
    
    auto engine = trtorch::ConvertGraphToTRTEngine(mod, "forward", input_ranges);
    return RunEngine(engine, inputs);
}

} // namespace util
} // namespace tests
} // namespace trtorch
