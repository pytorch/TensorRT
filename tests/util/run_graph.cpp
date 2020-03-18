#include "torch/csrc/jit/graph_executor.h"
#include "tests/util/util.h"

namespace trtorch {
namespace tests {
namespace util {

torch::jit::Stack CreateStack(std::vector<at::Tensor>&& list) {
    return torch::jit::Stack(
        std::make_move_iterator(list.begin()),
        std::make_move_iterator(list.end()));
}


std::vector<at::Tensor> RunGraph(std::shared_ptr<torch::jit::Graph>& g,
                                 core::conversion::GraphParams& params,
                                 std::vector<at::Tensor> inputs) {
    LOG_DEBUG("Running JIT version");
    std::vector<at::Tensor> inputs_;
    for (auto in : inputs) {
        inputs_.push_back(in.clone());
    }

    for (auto* in : g->inputs()) {
        const auto iter = params.find(in);
        if (iter != params.end()) {
            inputs_.push_back(iter->second.clone());
        }
    }

    torch::jit::GraphExecutor executor(g);
    auto stack = CreateStack(std::move(inputs_));
    
    executor.run(stack);
    std::vector<at::Tensor> outputs;
    for (size_t i = 0; i < g->outputs().size(); i++) {
        outputs.push_back(stack[i].toTensor());
    }
    
    return outputs;
}

} // namespace util
} // namespace tests
} // namespace trtorch
