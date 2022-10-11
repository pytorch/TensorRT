#include "core/runtime/runtime.h"
#include "cpp_api_test.h"

TEST_P(CppAPITests, ModuleAsEngineIsClose) {
  std::vector<at::Tensor> inputs;
  std::vector<torch::jit::IValue> inputs_ivalues;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    inputs.push_back(at::randint(5, input_shapes[i], {at::kCUDA}).to(input_types[i]));
    inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
  }

  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());
  auto trt_results = torch_tensorrt::tests::util::RunModuleForwardAsEngine(mod, inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(
      jit_results[0], trt_results[0].reshape_as(jit_results[0])));
}

#ifndef DISABLE_TEST_IN_CI

INSTANTIATE_TEST_SUITE_P(
    ModuleAsEngineForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        PathAndInput({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/efficientnet_b0_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/vit_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}})));
#endif
