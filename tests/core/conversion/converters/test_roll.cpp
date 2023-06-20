#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenRollConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[1, 0, 3, 7]]()
            %3 : int[] = prim::Constant[value=[0, 1, 2, 3]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {2, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRollShiftsNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[0, -3, -3]]()
            %3 : int[] = prim::Constant[value=[1, 2, 3]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenRollDimsNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%1 : Tensor):
            %2 : int[] = prim::Constant[value=[0, -3, -3]]()
            %3 : int[] = prim::Constant[value=[1, 2, -1]]()
            %4 : Tensor = aten::roll(%1, %2, %3)
            return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  // Run Pytorch
  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}