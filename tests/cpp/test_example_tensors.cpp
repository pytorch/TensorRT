#include "cpp_api_test.h"

TEST_P(CppAPITests, InputsFromTensors) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randn(input_shapes[i], {at::kCUDA}).to(input_types[i]);
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto inputs = std::vector<torch_tensorrt::Input>{trt_inputs_ivalues[0].toTensor()};
  auto spec = torch_tensorrt::ts::CompileSpec(inputs);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(PathAndInput({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5})));
