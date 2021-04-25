#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ReduceToCorrectly) {
  std::string source_graph = R"IR(
    graph(%x, %device, %dtype, %nb, %copy, %format):
        %out : Tensor = aten::to(%x, %device, %dtype, %nb, %copy, %format)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%x, %device, %dtype, %nb, %copy, %format):
        %out : Tensor = aten::to(%x, %dtype, %nb, %copy, %format)
        return (%out))IR";

  trtorch::core::util::logging::get_logger().set_reportable_log_level(trtorch::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  trtorch::core::lowering::passes::ReduceToOperation(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
