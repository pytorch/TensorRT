#include "module_test.h"

TEST_P(ModuleTests, ModuleAsEngineIsClose) {
  std::vector<at::Tensor> inputs;
  std::vector<torch::jit::IValue> inputs_ivalues;
  for (auto in_shape : input_shapes) {
    inputs.push_back(at::randint(5, in_shape, {at::kCUDA}));
    inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
  }

  torch::jit::IValue jit_results_ivalues = trtorch::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());
  auto trt_results = trtorch::tests::util::RunModuleForwardAsEngine(mod, inputs);

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

TEST_P(ModuleTests, ModuleToModuleIsClose) {
  std::vector<at::Tensor> inputs;
  std::vector<torch::jit::IValue> inputs_ivalues;
  for (auto in_shape : input_shapes) {
    inputs.push_back(at::randint(5, in_shape, {at::kCUDA}));
    inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
  }

  torch::jit::IValue jit_results_ivalues = trtorch::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());

  auto forward_graph = mod.get_method("forward");
  std::vector<c10::ArrayRef<int64_t>> input_ranges;
  for (auto in : inputs) {
    input_ranges.push_back(in.sizes());
  }

  auto engine = trtorch::ConvertGraphToTRTEngine(mod, "forward", input_ranges);
  auto trt_mod = trtorch::EmbedEngineInNewModule(engine);

  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-5));
}

INSTANTIATE_TEST_SUITE_P(
    ModuleAsEngineForwardIsCloseSuite,
    ModuleTests,
    testing::Values(
        PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}})));