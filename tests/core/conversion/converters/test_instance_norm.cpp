#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

// Tensor instance_norm(
//     const Tensor& input,
//     const c10::optional<Tensor>& weight_opt /* optional */,
//     const c10::optional<Tensor>& bias_opt /* optional */,
//     const c10::optional<Tensor>& running_mean_opt /* optional */,
//     const c10::optional<Tensor>& running_var_opt /* optional */,
//     bool use_input_stats, double momentum, double eps, bool cudnn_enabled) 
inline constexpr auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor?,
            %2 : Tensor?,
            %3 : Tensor?,
            %4 : Tensor?,
            %5 : bool):
        %9 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=0.10000000000000001]()
        %7 : float = prim::Constant[value=1.0000000000000001e-05]()
        %8 : Tensor = aten::instance_norm(%0, %1, %2, %3, %4, %5, %6, %7, %9)
        return (%8)
)IR";

TEST(Converters, ATenInstanceNormConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
  torch::jit::IValue weight, bias, mean, var; // NoneType
  bool use_input_stats = true;

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenInstanceNormAffineConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto weight = at::randn({in.size(1)}).to(at::kCUDA);
  auto bias = at::randn({in.size(1)}).to(at::kCUDA);

  torch::jit::IValue mean, var; // NoneType
  bool use_input_stats = true;

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}


TEST(Converters, ATenInstanceNormRunningStatsConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  torch::jit::IValue weight, bias; // NoneType

  auto mean = at::randn({in.size(1)}).to(at::kCUDA);
  auto var = at::randn({in.size(1)}).to(at::kCUDA);
  bool use_input_stats = false;

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
