#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, TileToRepeatCorrectly) {
  std::string source_graph = R"IR(
        graph(%input, %dim):
            %o : Tensor = aten::tile(%input, %dim)
            return (%o))IR";
  std::string target_graph = R"IR(
        graph(%input, %dim):
            %o : Tensor = aten::repeat(%input, %dim)
            return (%o))IR";
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::ReplaceTileWithRepeat(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
