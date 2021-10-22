#include "core/runtime/runtime.h"
#include "cpp_api_test.h"

TEST_P(CppAPITests, ModuleAsEngineIsClose) {
  std::vector<at::Tensor> inputs;
  std::vector<torch::jit::IValue> inputs_ivalues;
  for (auto in_shape : input_shapes) {
    inputs.push_back(at::randint(5, in_shape, {at::kCUDA}));
    inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
  }

  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());
  auto trt_results = torch_tensorrt::tests::util::RunModuleForwardAsEngine(mod, inputs);

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), threshold));
}

TEST_P(CppAPITests, ModuleToEngineToModuleIsClose) {
  std::vector<at::Tensor> inputs;
  std::vector<torch::jit::IValue> inputs_ivalues;
  for (auto in_shape : input_shapes) {
    inputs.push_back(at::randint(5, in_shape, {at::kCUDA}));
    inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
  }

  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());

  std::vector<c10::ArrayRef<int64_t>> input_ranges;
  for (auto in : inputs) {
    input_ranges.push_back(in.sizes());
  }

  auto compile_spec = torch_tensorrt::ts::CompileSpec({input_ranges});
  int device_id = 0;
  cudaGetDevice(&device_id);
  compile_spec.device.device_type = torch_tensorrt::Device::DeviceType::kGPU;
  compile_spec.device.gpu_id = device_id;
  auto engine = torch_tensorrt::ts::ConvertMethodToTRTEngine(mod, "forward", input_ranges);
  auto trt_mod = torch_tensorrt::ts::EmbedEngineInNewModule(engine, compile_spec.device);

  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), threshold));
}

INSTANTIATE_TEST_SUITE_P(
    ModuleAsEngineForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/resnet50_traced.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/mobilenet_v2_traced.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/resnet50_scripted.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/efficientnet_b0_scripted.jit.pt", {{1, 3, 224, 224}}, 2e-5}),
        PathAndInSize({"tests/modules/vit_scripted.jit.pt", {{1, 3, 224, 224}}, 8e-3})));
