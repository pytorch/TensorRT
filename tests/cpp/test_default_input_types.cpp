#include "cpp_api_test.h"

TEST_P(CppAPITests, InputsUseDefault) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = trtorch::CompileSpec::Input(input_shapes[0]);
  auto spec = trtorch::CompileSpec({in});
  spec.enabled_precisions.insert(trtorch::CompileSpec::DataType::kHalf);

  mod.to(torch::kHalf);

  auto trt_mod = trtorch::CompileGraph(mod, spec);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, 2e-5})));
