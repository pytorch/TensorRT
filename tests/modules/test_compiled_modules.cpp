#include "module_test.h"

TEST_P(ModuleTests, CompiledModuleIsClose) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  torch::jit::IValue jit_results_ivalues = trtorch::tests::util::RunModuleForward(mod, jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());

  auto trt_mod = trtorch::CompileGraph(mod, input_shapes);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());

  for (size_t i = 0; i < trt_results.size(); i++) {
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[i], trt_results[i].reshape_as(jit_results[i]), 2e-5));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    ModuleTests,
    testing::Values(
        PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}})));
