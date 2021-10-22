#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenLayerNormConvertsCorrectlyLast3DimsNoGammaBeta) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %gamma : None = prim::Constant()
        %beta : None = prim::Constant()
        %1: int = prim::Constant[value=3]()
        %2: int = prim::Constant[value=100]()
        %3: int = prim::Constant[value=100]()
        %4 : int[] = prim::ListConstruct(%1, %2, %3)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 3, 100, 100}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectlyLast3Dims) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %gamma: Float(3, 100, 100),
            %beta: Float(3, 100, 100)):
        %1: int = prim::Constant[value=3]()
        %2: int = prim::Constant[value=100]()
        %3: int = prim::Constant[value=100]()
        %4 : int[] = prim::ListConstruct(%1, %2, %3)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 3, 100, 100}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {3, 100, 100}, {at::kCUDA});
  auto beta = at::randint(1, 10, {3, 100, 100}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectlyLast2Dims) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %gamma : Float(100, 100),
            %beta : Float(100, 100)):
        %2: int = prim::Constant[value=100]()
        %3: int = prim::Constant[value=100]()
        %4 : int[] = prim::ListConstruct(%2, %3)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 3, 100, 100}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {100, 100}, {at::kCUDA});
  auto beta = at::randint(1, 10, {100, 100}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectlyLast1Dims) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %gamma: Float(100),
            %beta: Float(100)):
        %3: int = prim::Constant[value=100]()
        %4 : int[] = prim::ListConstruct(%3)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 3, 100, 100}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {100}, {at::kCUDA});
  auto beta = at::randint(1, 10, {100}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenLayerNormConvertsCorrectly3dInput1dNormalizedShape) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %gamma: Float(197, 768),
            %beta: Float(197, 768)):
        %1: int = prim::Constant[value=768]()
        %4 : int[] = prim::ListConstruct(%1)
        %7 : bool = prim::Constant[value=0]()
        %8 : float = prim::Constant[value=1.0000000000000001e-05]()
        %9 : Tensor = aten::layer_norm(%0, %4, %gamma, %beta, %8, %7)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 197, 768}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {768}, {at::kCUDA});
  auto beta = at::randint(1, 10, {768}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
