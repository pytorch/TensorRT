#include <string>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

#ifndef DISABLE_TEST_IN_CI

TEST(CppAPITest, CanRunMultipleEngines) {
  torch::jit::script::Module mod1;
  torch::jit::script::Module mod2;
  try {
    mod1 = torch::jit::load("tests/modules/resnet50_traced.jit.pt");
    mod2 = torch::jit::load("tests/modules/resnet18_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};

  std::vector<torch::jit::IValue> jit1_inputs_ivalues;
  std::vector<torch::jit::IValue> trt1_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit1_inputs_ivalues.push_back(in.clone());
    trt1_inputs_ivalues.push_back(in.clone());
  }

  std::vector<torch::jit::IValue> jit2_inputs_ivalues;
  std::vector<torch::jit::IValue> trt2_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit2_inputs_ivalues.push_back(in.clone());
    trt2_inputs_ivalues.push_back(in.clone());
  }

  torch::jit::IValue jit1_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod1, jit1_inputs_ivalues);
  std::vector<at::Tensor> jit1_results;
  jit1_results.push_back(jit1_results_ivalues.toTensor());

  torch::jit::IValue jit2_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod2, jit2_inputs_ivalues);
  std::vector<at::Tensor> jit2_results;
  jit2_results.push_back(jit2_results_ivalues.toTensor());

  auto trt_mod1 = torch_tensorrt::ts::compile(mod1, input_shapes);
  torch::jit::IValue trt1_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(trt_mod1, trt1_inputs_ivalues);
  std::vector<at::Tensor> trt1_results;
  trt1_results.push_back(trt1_results_ivalues.toTensor());

  auto trt_mod2 = torch_tensorrt::ts::compile(mod2, input_shapes);
  torch::jit::IValue trt2_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(trt_mod2, trt2_inputs_ivalues);
  std::vector<at::Tensor> trt2_results;
  trt2_results.push_back(trt2_results_ivalues.toTensor());

  for (size_t i = 0; i < trt1_results.size(); i++) {
    ASSERT_TRUE(
        torch_tensorrt::tests::util::almostEqual(jit1_results[i], trt1_results[i].reshape_as(jit1_results[i]), 2e-5));
  }

  for (size_t i = 0; i < trt2_results.size(); i++) {
    ASSERT_TRUE(
        torch_tensorrt::tests::util::almostEqual(jit2_results[i], trt2_results[i].reshape_as(jit2_results[i]), 2e-5));
  }
}
#endif
