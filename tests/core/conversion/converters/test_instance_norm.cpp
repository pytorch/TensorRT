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
constexpr auto graph = R"IR(
      graph(%input.1 : Tensor,
            %weight.1 : Tensor?,
            %bias.1 : Tensor?,
            %running_mean.1 : Tensor?,
            %running_var.1 : Tensor?,
            %use_input_stats.1 : bool):
        %cudnn_enabled.1 : bool = prim::Constant[value=1]()
        %momentum.1 : float = prim::Constant[value=0.10000000000000001]()
        %eps.1 : float = prim::Constant[value=1.0000000000000001e-05]()
        %4 : Tensor = aten::instance_norm(%input.1,
          %weight.1, %bias.1,
          %running_mean.1, %running_var.1,
          %use_input_stats.1, %momentum.1, %eps.1, %cudnn_enabled.1)
        return (%4)
)IR";

TEST(Converters, ATenInstanceNormConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
  torch::jit::IValue weight, bias, mean, var; // NoneType
  // https://github.com/pytorch/pytorch/blob/79693bb86a3f601a5c0d3da52d99acec95bb48c1/torch/nn/modules/instancenorm.py#L59
  const bool use_input_stats = true;

  auto trt_in = at::clone(in);
  torch::jit::IValue trt_weight, trt_bias, trt_mean, trt_var;

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(
      g->inputs(), {trt_weight, trt_bias, trt_mean, trt_var, use_input_stats});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenInstanceNormAffineConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto weight = at::randn({in.size(1)}).to(at::kCUDA);
  auto bias = at::randn({in.size(1)}).to(at::kCUDA);

  torch::jit::IValue mean, var; // NoneType
  const bool use_input_stats = true;

  auto trt_in = at::clone(in);
  auto trt_weight = at::clone(weight);
  auto trt_bias = at::clone(bias);
  torch::jit::IValue trt_mean, trt_var;

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(
      g->inputs(), {trt_weight, trt_bias, trt_mean, trt_var, use_input_stats});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenInstanceNormRunningStatsConvertsCorrectly) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randn({1, 5, 5, 5}, {at::kCUDA});

  torch::jit::IValue weight, bias;
  auto mean = at::zeros({in.size(1)}, {at::kCUDA});
  auto var = at::ones({in.size(1)}, {at::kCUDA});
  const bool use_input_stats = false;

  auto trt_in = at::clone(in);
  torch::jit::IValue trt_weight, trt_bias;
  auto trt_mean = at::clone(mean);
  auto trt_var = at::clone(var);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {weight, bias, mean, var, use_input_stats});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(
      g->inputs(), {trt_weight, trt_bias, trt_mean, trt_var, use_input_stats});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
