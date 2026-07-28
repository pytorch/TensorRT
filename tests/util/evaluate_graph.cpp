#include "tests/util/util.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

#include "core/conversion/conversion.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/evaluators/evaluators.h"
#include "core/conversion/var/Var.h"
#include "core/util/jit_util.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace tests {
namespace util {

std::vector<torch::jit::IValue> EvaluateGraph(const torch::jit::Block* b, std::vector<torch::jit::IValue> inputs) {
  LOG_DEBUG("Running Torch-TensorRT Version");

  core::conversion::ConversionCtx* ctx = new core::conversion::ConversionCtx({});

  TORCHTRT_CHECK(inputs.size() == b->inputs().size(), "Amount of provided inputs do not match number of graph inputs");
  for (size_t i = 0; i < inputs.size(); i++) {
    ctx->AssociateValueAndIValue(b->inputs()[i], inputs[i]);
  }
  LOG_DEBUG("Checking nodes");
  for (const auto n : b->nodes()) {
    TORCHTRT_CHECK(
        core::conversion::evaluators::shouldEvalAtConversionTime(n),
        "Test graph contains non evaluatable nodes: " << *n);
    auto eval = core::conversion::EvaluateNode(ctx, n);
    if (eval) {
      if (eval.value().isTuple() && n->outputs().size() > 1) {
        auto eval_list = eval.value().toTuple();
        for (size_t i = 0; i < eval_list->elements().size(); i++) {
          auto eval_output = eval_list.get()->elements()[i];
          LOG_DEBUG(
              ctx->logger,
              "Found the evaluated value(s) to be " << eval_output
                                                    << " for node: " << torch_tensorrt::core::util::node_info(n));
          ctx->AssociateValueAndIValue(n->output(i), eval_output);
        }
      } else if (!eval.value().isTensor()) {
        LOG_DEBUG("Found the value to be: " << eval.value());
        ctx->AssociateValueAndIValue(n->output(0), eval.value());
      } else {
        LOG_DEBUG("Found the value to be a tensor (shape " << eval.value().toTensor().sizes() << ')');
        ctx->AssociateValueAndIValue(n->output(0), eval.value());
      }
    }
  }

  std::vector<torch::jit::IValue> outputs;
  for (auto o : b->outputs()) {
    auto it = ctx->evaluated_value_map.find(o);
    TORCHTRT_CHECK(
        it != ctx->evaluated_value_map.end(),
        "No corresponding IValue found for TorchScript Value: " << o->debugName());
    outputs.push_back(it->second);
  }

  delete ctx;
  return outputs;
}

std::vector<torch::jit::IValue> EvaluateGraphJIT(
    std::shared_ptr<torch::jit::Graph>& g,
    std::vector<torch::jit::IValue> inputs) {
  LOG_DEBUG("Running JIT version");

  torch::jit::GraphExecutor executor(g, "");
  auto stack = torch::jit::Stack();
  for (auto& i : inputs) {
    torch::jit::push(stack, i);
  }

  executor.run(stack);
  std::vector<torch::jit::IValue> outputs;
  for (size_t i = 0; i < g->outputs().size(); i++) {
    outputs.push_back(stack[i]);
  }

  return outputs;
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
