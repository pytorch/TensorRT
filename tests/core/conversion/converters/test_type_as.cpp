#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenTypeAsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %a : int = prim::Constant[value=1]()
        %4 : Tensor = aten::add(%0, %2, %a)
        %5 : Tensor = aten::gt(%1, %a)
        %6 : Tensor = aten::type_as(%4, %5)
        return (%6, %5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in1 = at::randint(1, 3, {10}, {at::kCUDA});
  auto in2 = at::randint(1, 3, {10}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1, in2});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1, in2});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}