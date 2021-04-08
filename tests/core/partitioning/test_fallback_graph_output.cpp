#include <string>
#include <unordered_set>
#include "gtest/gtest.h"
#include "torch/script.h"
#include "core/compiler.h"
#include "tests/util/util.h"


TEST(Partitioning, StitchSegmentedBlockCorrectly) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_base_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{3, 3, 16, 16}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::core::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(Partitioning, StitchSegmentedBlockCorrectlyEdge) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_edge_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{3, 3, 16, 16}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
  auto in = at::randint(5, in_shape, {at::kCUDA});
  jit_inputs_ivalues.push_back(in.clone());
  trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::core::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

