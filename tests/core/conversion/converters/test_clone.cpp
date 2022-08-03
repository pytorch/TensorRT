#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenCloneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor = aten::relu(%0)
        %2 : None = prim::Constant()
        %3 : Tensor = aten::clone(%1, %2)
        %4 : Tensor = aten::relu(%3)
        %5 : int = prim::Constant[value=1]()
        %6 : Tensor = aten::add(%1, %4, %5)
        return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
