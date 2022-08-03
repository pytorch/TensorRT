#include "cpp_api_test.h"

// Following test is applicable for multi-gpu environment only
TEST_P(CppAPITests, CompiledModuleIsClose) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randint(5, input_shapes[i], {at::kCUDA}).to(input_types[i]);
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());

  auto trt_mod = torch_tensorrt::ts::compile(mod, input_shapes);

  // Deliberately changing the device ID. torch_tensorrt runtime should correct the Device ID internally
  torch_tensorrt::set_device(1);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());

  for (size_t i = 0; i < trt_results.size(); i++) {
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
        jit_results[i], trt_results[i].reshape_as(jit_results[i]).to(torch::Device("cuda:0")), 2e-5));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(PathAndInput({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5})));
