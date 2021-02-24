#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenStackPureTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor):
        %2 : Tensor[] = prim::ListConstruct(%0, %1)
        %3 : int = prim::Constant[value=3]()
        %4 : Tensor = aten::stack(%2, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});
  auto in2 = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1, in2});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1, in2});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenStackDiffTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4:16, 4:4, 4:1)):
        %2 : Tensor[] = prim::ListConstruct(%0, %1)
        %3 : int = prim::Constant[value=1]()
        %4 : Tensor = aten::stack(%2, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});
  auto in2 = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {in2});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {in2});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}