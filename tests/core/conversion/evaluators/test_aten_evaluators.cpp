#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Evaluators, DivIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=9]()
        %2 : int = prim::Constant[value=4]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, DivFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : float = prim::Constant[value=9.1]()
        %2 : float = prim::Constant[value=4.2]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}