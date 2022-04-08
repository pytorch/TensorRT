#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenBatchNormConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5, strides=[1]),
            %2: Float(5, strides=[1]),
            %3: Float(5, strides=[1]),
            %4: Float(5, strides=[1])):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {5}, {at::kCUDA});
  auto beta = at::randint(1, 10, {5}, {at::kCUDA});
  auto mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto var = at::randint(1, 10, {5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta, mean, var});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta, mean, var});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenBatchNormAffineFalseConvertsCorrectly) {
  // BatchNorm(ch, affine=False)
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %3: Float(5, strides=[1]),
            %4: Float(5, strides=[1])):
        %1 : None = prim::Constant()
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %1, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto var = at::randint(1, 10, {5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {mean, var});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {mean, var});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenBatchNorm1DConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5, strides=[1]),
            %2: Float(5, strides=[1]),
            %3: Float(5, strides=[1]),
            %4: Float(5, strides=[1])):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Test 2D tensor, which is valid shape for BatchNorm1D ops.
  auto in = at::randint(1, 10, {1, 5}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {5}, {at::kCUDA});
  auto beta = at::randint(1, 10, {5}, {at::kCUDA});
  auto mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto var = at::randint(1, 10, {5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta, mean, var});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta, mean, var});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenBatchNormShouldUnpackConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5, strides=[1]),
            %2: Float(5, strides=[1]),
            %3: Float(5, strides=[1]),
            %4: Float(5, strides=[1])):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {2, 5, 5, 5}, {at::kCUDA});
  auto gamma = at::randint(1, 10, {5}, {at::kCUDA});
  auto beta = at::randint(1, 10, {5}, {at::kCUDA});
  auto mean = at::randint(1, 10, {5}, {at::kCUDA});
  auto var = at::randint(1, 10, {5}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_gamma = at::clone(gamma);
  auto trt_beta = at::clone(beta);
  auto trt_mean = at::clone(mean);
  auto trt_var = at::clone(var);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {gamma, beta, mean, var});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_gamma, trt_beta, trt_mean, trt_var});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
