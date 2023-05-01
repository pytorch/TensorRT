#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, AutocastLongInputs) {
  std::string source_graph = R"IR(
    graph(%long_0 : Tensor, %long_1 : Tensor):
      %res  : Tensor = aten::add(%long_0, %long_1)
      return (%res))IR";
  std::string target_graph = R"IR(
    graph(%long_0 : Tensor, %long_1 : Tensor):
      %3 : bool = prim::Constant[value=0]()
      %4 : Device = prim::Constant[value="cuda:0"]()
      %5 : NoneType = prim::Constant()
      %6 : int = prim::Constant[value=4]()
      %7 : Tensor = aten::to[to_compile=0](%long_0, %4, %6, %3, %3, %5)
      %8 : int = prim::Constant[value=4]()
      %9 : Tensor = aten::to[to_compile=0](%long_1, %4, %8, %3, %3, %5)
      %2 : Tensor = aten::add(%7, %9)
      return (%2))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>> type_map;
  type_map[sg->inputs()[0]] = at::kLong;
  type_map[sg->inputs()[1]] = at::kLong;
  torch_tensorrt::core::lowering::AutocastLongInputs(sg, type_map, "cuda:0");
  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  ASSERT_TRUE(sg->nodes().front()->kind() == torch::jit::prim::Constant); // confirm constants are added before casts
  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
