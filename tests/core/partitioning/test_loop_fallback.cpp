#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"

TEST(Partitioning, CheckLoopFallbackEvalCompilesCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/loop_fallback_eval_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 10}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<torch_tensorrt::core::ir::Input> input_ranges{torch_tensorrt::core::ir::Input({1, 10})};
  torch_tensorrt::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(Partitioning, CheckLoopFallbackNoEvalCompilesCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/loop_fallback_no_eval_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 10}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<torch_tensorrt::core::ir::Input> input_ranges{torch_tensorrt::core::ir::Input({1, 10})};
  torch_tensorrt::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
