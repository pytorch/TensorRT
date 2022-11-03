#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

TEST(CppAPITest, ResNet18DynamicBatchFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet18_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}, {4, 3, 224, 224}, {8, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  auto in_bs1 = at::randint(5, input_shapes[0], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_bs1.clone());
  trt_inputs_ivalues.push_back(in_bs1.clone());

  std::vector<torch_tensorrt::Input> inputs;
  inputs.push_back(torch_tensorrt::Input(input_shapes[0], input_shapes[1], input_shapes[2]));
  torch_tensorrt::ts::CompileSpec cfg(inputs);
  cfg.torch_executed_ops.push_back("aten::add");

  auto jit_results_bs1 = mod.forward(jit_inputs_ivalues).toTensor();
  // Compile and build the hybrid graph with dynamic shapes
  auto trt_mod = torch_tensorrt::ts::compile(mod, cfg);
  auto trt_results_bs1 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_bs1, trt_results_bs1));
  jit_inputs_ivalues.clear();
  trt_inputs_ivalues.clear();

  // Run with batch size of 4
  auto in_bs4 = at::randint(5, input_shapes[1], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_bs4.clone());
  trt_inputs_ivalues.push_back(in_bs4.clone());

  auto jit_results_bs4 = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_results_bs4 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_bs4, trt_results_bs4));
  jit_inputs_ivalues.clear();
  trt_inputs_ivalues.clear();

  // Run with batch size of 8
  auto in_bs8 = at::randint(5, input_shapes[2], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_bs8.clone());
  trt_inputs_ivalues.push_back(in_bs8.clone());

  auto jit_results_bs8 = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_results_bs8 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_bs8, trt_results_bs8));
}

TEST(CppAPITest, ResNet18DynamicShapeFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet18_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 64, 64}, {1, 3, 128, 128}, {1, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  auto in_64 = at::randint(5, input_shapes[0], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_64.clone());
  trt_inputs_ivalues.push_back(in_64.clone());

  std::vector<torch_tensorrt::Input> inputs;
  inputs.push_back(torch_tensorrt::Input(input_shapes[0], input_shapes[1], input_shapes[2]));
  torch_tensorrt::ts::CompileSpec cfg(inputs);
  cfg.torch_executed_ops.push_back("aten::add");

  auto jit_results_64 = mod.forward(jit_inputs_ivalues).toTensor();
  // Compile and build the hybrid graph with dynamic shapes
  auto trt_mod = torch_tensorrt::ts::compile(mod, cfg);
  auto trt_results_64 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_64, trt_results_64));
  jit_inputs_ivalues.clear();
  trt_inputs_ivalues.clear();

  // Run with input resolution of (1, 3, 128, 128)
  auto in_128 = at::randint(5, input_shapes[1], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_128.clone());
  trt_inputs_ivalues.push_back(in_128.clone());

  auto jit_results_128 = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_results_128 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_128, trt_results_128));
  jit_inputs_ivalues.clear();
  trt_inputs_ivalues.clear();

  // Run with input resolution of (1, 3, 256, 256)
  auto in_256 = at::randint(5, input_shapes[2], {at::kCUDA});
  jit_inputs_ivalues.push_back(in_256.clone());
  trt_inputs_ivalues.push_back(in_256.clone());

  auto jit_results_256 = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_results_256 = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results_256, trt_results_256));
}
