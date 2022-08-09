#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenCopyConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %0.1 : Tensor = aten::relu(%0)
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=3]()
        %3 : int = prim::Constant[value=10]()
        %4 : int = prim::Constant[value=10]()
        %5 : int[] = prim::ListConstruct(%1, %2, %3, %4)
        %6 : None = prim::Constant()
        %7 : Device = prim::Constant[value="cuda"]()
        %8 : Tensor = aten::ones(%5, %6, %6, %7, %6)
        %9 : bool = prim::Constant[value=0]()
        %10 : Tensor = aten::copy_(%8, %0.1, %9)
        %11 : Tensor = aten::relu(%10)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::add(%0.1, %11, %12)
        return (%13))IR";

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
