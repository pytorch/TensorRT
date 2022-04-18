#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenReLUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::relu(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenSigmoidConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::sigmoid(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 5e-6));
}

TEST(Converters, ATenTanhConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::tanh(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 7e-6));
}

// TODO: Seems like the IR parser is not handling negative numbers well, need to
// follow up with the PyTorch Team
TEST(Converters, ATenHardTanhConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : float = prim::Constant[value=-1.0]()
      %2 : float = prim::Constant[value=1.0]()
      %3 : Tensor = aten::hardtanh(%0, %1, %2)
      return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenHardTanhCustomRangeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=0.0]()
        %2 : float = prim::Constant[value=6.0]()
        %3 : Tensor = aten::hardtanh(%0, %1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPReLUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(1, strides=[1])):
        %3 : Tensor = aten::prelu(%0, %1)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto slope = at::randint(-5, 5, {1}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {slope});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {slope});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPReLUChannelAlignConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1.1 : Float(3,strides=[1]),
            %1 : Float(8, 3, 5, 5, strides=[45, 15, 5, 1]),
            %2 : Float(8)):
        %0.1 : Tensor = aten::prelu(%0, %1.1)
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : Tensor = aten::_convolution(%0.1, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});
  auto w_prelu = at::randint(1, 10, {3}, {at::kCUDA});
  auto w = at::randint(1, 10, {8, 3, 5, 5}, {at::kCUDA});
  auto b = at::randint(1, 10, {8}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w_prelu = at::clone(w_prelu);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w_prelu, jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w_prelu = at::clone(w_prelu);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w_prelu, trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenPReLUMultiChannelConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(10, strides=[1])):
        %3 : Tensor = aten::prelu(%0, %1)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {1, 10, 1, 1}, {at::kCUDA});
  auto slope = at::randint(-5, 5, {10}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {slope});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {slope});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenLeakyReluConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=0.15]()
        %2 : Tensor = aten::leaky_relu(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenEluConvertsCorrectly) {
  const auto graph = R"IR(
       graph(%x.1 : Tensor):
        %2 : float = prim::Constant[value=1.]()
        %3 : int = prim::Constant[value=1]()
        %result.2 : Tensor = aten::elu(%x.1, %2, %3, %3)
        return (%result.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(-5, 5, {1, 10, 1, 1}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

#ifndef DISABLE_TEST_IN_CI
TEST(Converters, ATenGELUConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::gelu(%0)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});

  // Lower aten::gelu to pointwise operators using Fast approximation
  // Gelu(x) = 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
  torch_tensorrt::core::lowering::passes::ReduceGelu(g);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  // NOTE: The official tensorrt plugin applies the Gelu activation x * Phi(x), where Phi is the Gaussian cdf,
  // approximated by: 0.5 * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x^3))) and the pytorch uses
  // c10::cuda::compat::normcdf to compute Phi(x). So there's a difference here and therefore the threshold is slightly
  // higher than other ops. One in ten runs will give you an out of normal threshold result

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 5e-2));
}
#endif
