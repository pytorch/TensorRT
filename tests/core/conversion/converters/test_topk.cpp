#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenTopKConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=20]()
        %2 : int = prim::Constant[value=-1]()
        %3 : bool = prim::Constant[value=1]()
        %4 : bool = prim::Constant[value=1]()
        %5 : Tensor, %6 : Tensor = aten::topk(%0, %1, %2, %3, %4)
        return (%5, %6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({10, 10, 100}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[1], trt_results[1].reshape_as(jit_results[1]), 2e-6));
}
