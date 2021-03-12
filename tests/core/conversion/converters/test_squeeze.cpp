#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenSqueezeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : Tensor = aten::squeeze(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {2, 1, 3, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenSqueezeDontNeedSqueezeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %2.1 : Tensor = aten::add(%0, %1, %2)
        %3 : Tensor = aten::squeeze(%2.1, %2)
        %4 : Tensor = aten::add(%3, %1, %2)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {2, 3, 3}, {at::kCUDA});
  auto in_add = at::randint(1, 10, {2, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_in_add = at::clone(in_add);

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in, jit_in_add});

  auto trt_in = at::clone(jit_in);
  auto trt_in_add = at::clone(jit_in_add);

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in, trt_in_add});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}