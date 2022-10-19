#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

TEST(CppAPITest, ResNet50DynamicFallbackGraphCorrectly) {
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
  auto in = at::randint(5, input_shapes[0], {at::kCUDA});
  jit_inputs_ivalues.push_back(in.clone());
  trt_inputs_ivalues.push_back(in.clone());

  std::vector<torch_tensorrt::Input> inputs;
  inputs.push_back(torch_tensorrt::Input(input_shapes[0], input_shapes[1], input_shapes[2]));
  torch_tensorrt::ts::CompileSpec cfg(inputs);
  cfg.torch_executed_ops.push_back("aten::add");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = torch_tensorrt::ts::compile(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results, trt_results));
}
