#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenMaxDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor, %5 : Tensor = aten::max(%x.1, %2, %3)
      return (%4, %5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1].reshape_as(jit_results[1]), 2e-6));
}

TEST(Converters, ATenMaxDimIntInputConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor, %5 : Tensor = aten::max(%x.1, %2, %3)
      return (%4, %5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1], 2e-6));
}

TEST(Converters, ATenMinDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor, %5 : Tensor = aten::min(%x.1, %2, %3)
      return (%4, %5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[1], trt_results[1].reshape_as(jit_results[1]), 2e-6));
}

TEST(Converters, ATenArgMaxConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor = aten::argmax(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenArgMaxIntInputConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor = aten::argmax(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenArgMaxKeepdimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=1]()
      %4 : Tensor = aten::argmax(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenArgMinConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=0]()
      %4 : Tensor = aten::argmin(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenArgMinKeepdimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=1]()
      %4 : Tensor = aten::argmin(%x.1, %2, %3)
      return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::rand({2, 3, 5, 5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
