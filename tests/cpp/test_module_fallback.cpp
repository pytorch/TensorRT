#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "trtorch/trtorch.h"

TEST(CppAPITests, LowerResNetModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet18_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::CompileSpec::Input> input_ranges{
      trtorch::CompileSpec::Input(std::vector<int64_t>({1, 3, 224, 224}))};
  trtorch::CompileSpec cfg(input_ranges);
  cfg.torch_fallback.enabled = true;
  cfg.torch_fallback.forced_fallback_modules.push_back("torchvision.models.resnet.BasicBlock");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(CppAPITests, LowerAndPartitionMobileNetModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::CompileSpec::Input> input_ranges{
      trtorch::CompileSpec::Input(std::vector<int64_t>({1, 3, 224, 224}))};
  trtorch::CompileSpec cfg(input_ranges);
  cfg.torch_fallback.enabled = true;
  cfg.torch_fallback.min_block_size = 5;
  cfg.torch_fallback.forced_fallback_modules.push_back("torchvision.models.mobilenetv2.ConvBNActivation");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::CompileGraph(mod, cfg);

  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
