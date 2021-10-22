#include "cpp_api_test.h"

TEST_P(CppAPITests, InputsFromTensors) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto spec = torch_tensorrt::ts::CompileSpec({trt_inputs_ivalues[0].toTensor()});

  auto trt_mod = torch_tensorrt::ts::CompileModule(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, 2e-5})));
