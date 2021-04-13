#include <iostream>
#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenReplication_pad1dTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %2 : Tensor = aten::replication_pad1d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad1dRightZeroTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 0]]()
        %2 : Tensor = aten::replication_pad1d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad1dTensorConvertsCorrectlyWithDynamic) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %2 : Tensor = aten::replication_pad1d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad2dTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %2 : Tensor = aten::replication_pad2d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad2dRightBottomZeroTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 0, 2, 0]]()
        %2 : Tensor = aten::replication_pad2d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad2dTensorConvertsCorrectlyWithDynamic) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %2 : Tensor = aten::replication_pad2d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad3dTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3, 1, 4]]()
        %2 : Tensor = aten::replication_pad3d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad3dRightBottomBackZeroTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 0, 2, 0, 1, 0]]()
        %2 : Tensor = aten::replication_pad3d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenReplication_pad3dTensorConvertsCorrectlyWithDynamic) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3, 1, 4]]()
        %2 : Tensor = aten::replication_pad3d(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in1 = at::randint(1, 10, {1, 3, 4, 5, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}