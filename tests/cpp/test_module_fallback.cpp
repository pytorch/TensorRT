#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "trtorch/trtorch.h"

TEST(CppAPITest, ResNetModuleFallbacksCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet18_scripted.jit.pt");
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

  trtorch::CompileSpec cfg(input_shapes);
  cfg.torch_fallback.enabled = true;
  cfg.torch_fallback.forced_fallback_modules.push_back("torchvision.models.resnet.BasicBlock");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(CppAPITest, MobileNetModuleFallbacksCorrectlyWithOneEngine) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_scripted.jit.pt");
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

  trtorch::CompileSpec cfg(input_shapes);
  cfg.torch_fallback.enabled = true;
  cfg.torch_fallback.min_block_size = 5;
  cfg.torch_fallback.forced_fallback_modules.push_back("torchvision.models.mobilenetv2.ConvBNActivation");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::CompileGraph(mod, cfg);

  auto g = trt_mod.get_method("forward").graph();
  auto nodes = g->block()->nodes();
  std::size_t trt_count = 0;
  for (const auto n : nodes) {
    if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
      trt_count++;
    }
  }
  ASSERT_TRUE(trt_count == 1);

  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
