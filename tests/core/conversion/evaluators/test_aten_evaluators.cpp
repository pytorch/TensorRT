#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

TEST(Evaluators, ATenIsFloatingPointEvaluatesTrueCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : bool = aten::is_floating_point(%0)
        return (%1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3, 3, 3}, {at::kCUDA}).to(torch::kF32);
  auto in_trt = in.clone();

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in_trt});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, ATenIsFloatingPointEvaluatesFalseCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : bool = aten::is_floating_point(%0)
        return (%1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {1, 3, 3, 3}, {at::kCUDA}).to(torch::kI8);
  auto in_trt = in.clone();

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in_trt});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}
