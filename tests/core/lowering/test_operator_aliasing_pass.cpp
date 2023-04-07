#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, LoweringTrueDivideCorrectly) {
  std::string source_graph = R"IR(
    graph(%s, %o):
      %2 = aten::true_divide(%s, %o)
      return (%2))IR";
  std::string target_graph = R"IR(
    graph(%s, %o):
      %2 = aten::div(%s, %o)
      return (%2))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::AliasOperators(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, LoweringMultiplyCorrectly) {
  std::string source_graph = R"IR(
    graph(%s, %o):
      %2 = aten::multiply(%s, %o)
      return (%2))IR";
  std::string target_graph = R"IR(
    graph(%s, %o):
      %2 = aten::mul(%s, %o)
      return (%2))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::AliasOperators(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, LoweringPadToConstantPadNdCorrectly) {
  std::string source_graph = R"IR(
        graph(%input, %pad, %value):
            %mode : str = prim::Constant[value="constant"]()
            %o : Tensor = aten::pad(%input, %pad, %mode, %value)
            return (%o))IR";
  std::string target_graph = R"IR(
        graph(%input, %pad, %value):
            %o : Tensor = aten::constant_pad_nd(%input, %pad, %value)
            return (%o))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::AliasOperators(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, LoweringPadToConstantPadNdNoneInputCorrectly) {
  std::string source_graph = R"IR(
        graph(%input, %pad):
            %none : NoneType = prim::Constant()
            %mode : str = prim::Constant[value="constant"]()
            %o : Tensor = aten::pad(%input, %pad, %mode, %none)
            return (%o))IR";
  std::string target_graph = R"IR(
        graph(%input, %pad, %value):
            %none : NoneType = prim::Constant()
            %zero : int = prim::Constant[value=0]()
            %o : Tensor = aten::constant_pad_nd(%input, %pad, %zero)
            return (%o))IR";
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::AliasOperators(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
