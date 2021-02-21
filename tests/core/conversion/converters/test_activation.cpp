#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenReLUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::relu(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenSigmoidConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::sigmoid(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenTanhConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::tanh(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

// TODO: Seems like the IR parser is not handling negative numbers well, need to
// follow up with the PyTorch Team
// TEST(Converters, ATenHardTanhConvertsCorrectly) {
//     const auto graph = R"IR(
//       graph(%0 : Tensor):
//         %1 : float = prim::Constant[value=-1.0]()
//         %2 : float = prim::Constant[value=1.0]()
//         %3 : Tensor = aten::hardtanh(%0, %1, %2)
//         return (%3))IR";

//     auto g = std::make_shared<torch::jit::Graph>();
//     torch::jit::script::parseIR(graph, &*g);

//     auto in = at::randint(-5, 5, {5}, {at::kCUDA});
//     auto params = trtorch::core::conversion::get_named_params(g->inputs(),
//     {}); auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

//     in = at::clone(in);
//     params = trtorch::core::conversion::get_named_params(g->inputs(), {});
//     auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

//     ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0],
//     trt_results[0], 2e-6));
// }

TEST(Converters, ATenHardTanhCustomRangeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=0.0]()
        %2 : float = prim::Constant[value=6.0]()
        %3 : Tensor = aten::hardtanh(%0, %1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPReLUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(1:1)):
        %3 : Tensor = aten::prelu(%0, %1)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto slope = at::randint(-5, 5, {1}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {slope});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {slope});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPReLUMultiChannelConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(10:1)):
        %3 : Tensor = aten::prelu(%0, %1)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {1, 10, 1, 1}, {at::kCUDA});
  auto slope = at::randint(-5, 5, {10}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {slope});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {slope});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenLeakyReluConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=0.15]()
        %2 : Tensor = aten::leaky_relu(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenGELUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::gelu(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
  // The official tensorrt plugin applies the Gelu activation x * Phi(x), where Phi is the Gaussian cdf, approximated
  // by: 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x^3))) and the pytorch uses c10::cuda::compat::normcdf to
  // compute Phi(x). So there's a difference here.
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-4));
}