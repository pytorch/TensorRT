#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RemoveDetachCorrectly) {
  std::string source_graph = R"IR(
    graph(%input):
      %2 = aten::detach(%input)
      %3 = aten::sin(%2)
      return (%3))IR";
  std::string target_graph = R"IR(
    graph(%input):
      %3 = aten::sin(%input)
      return (%3))IR";

  trtorch::core::util::logging::get_logger().set_reportable_log_level(trtorch::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  trtorch::core::lowering::passes::RemoveNOPs(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}