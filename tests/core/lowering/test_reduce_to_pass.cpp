#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ReduceToDtypeLayoutCorrectly) {
  std::string source_graph = R"IR(
    graph(%x, %dtype, %layout, %device, %pm, %nb, %copy, %format):
        %out : Tensor = aten::to(%x, %dtype, %layout, %device, %pm, %nb, %copy, %format)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%x, %dtype, %layout, %device, %pm, %nb, %copy, %format):
        %out : Tensor = aten::to(%x, %device, %dtype, %nb, %copy, %format)
        return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceToOperation(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, ReduceAtenTypeAsCorrectly) {
  std::string source_graph = R"IR(
    graph(%input, %other):
        %out : Tensor = aten::type_as(%input, %other)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%input, %other):
        %5 : bool = prim::Constant[value=0]()
        %6 : None = prim::Constant()
        %out : Tensor = aten::to(%input, %other, %5, %5, %6)
        return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReduceToOperation(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
