#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RemoveToLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %6 : None = prim::Constant()
      %4 : bool = prim::Constant[value=0]()
      %3 : int = prim::Constant[value=5]() # experiments/test.py:8:17
      %y.1 : Tensor = aten::to(%x.1, %3, %4, %4, %6)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  trtorch::core::util::logging::get_logger().set_reportable_log_level(trtorch::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  trtorch::core::lowering::passes::RemoveNOPs(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}