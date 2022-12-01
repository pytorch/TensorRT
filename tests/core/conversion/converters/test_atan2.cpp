#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

TEST(Converters, ATenAtan2ConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%x.0 : Tensor, %x.1 : Tensor):
        %2 : Tensor = aten::atan2(%x.0, %x.1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Resize range to [-1, 1] to span multiple quadrants
  auto in_0 = -2 * at::rand({2, 3, 5, 5}, {at::kCUDA}) + 1;
  auto in_1 = -2 * at::rand({2, 3, 5, 5}, {at::kCUDA}) + 1;

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenAtan2ManagesPosInfCorrectly) {
  const auto graph = R"IR(
      graph(%x.0 : Tensor, %x.1 : Tensor):
        %2 : Tensor = aten::atan2(%x.0, %x.1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Expecting PI/2
  auto in_0 = at::ones({4, 1, 7, 8}, {at::kCUDA});
  auto in_1 = at::zeros({4, 1, 7, 8}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenAtan2ManagesNegInfCorrectly) {
  const auto graph = R"IR(
      graph(%x.0 : Tensor, %x.1 : Tensor):
        %2 : Tensor = aten::atan2(%x.0, %x.1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Expecting -PI/2
  auto in_0 = -1 * at::ones({4, 1, 7, 8}, {at::kCUDA});
  auto in_1 = at::zeros({4, 1, 7, 8}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}