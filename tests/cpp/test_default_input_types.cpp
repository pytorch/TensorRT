#include "cpp_api_test.h"
#include "torch_tensorrt/logging.h"

TEST_P(CppAPITests, InputsUseDefaultFP32) {
  torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto in = torch_tensorrt::Input(input_shapes[0]);
  auto spec = torch_tensorrt::ts::CompileSpec({in});
  spec.enabled_precisions.insert(torch_tensorrt::DataType::kHalf);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP32
}

TEST_P(CppAPITests, InputsUseDefaultFP16) {
  torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = torch_tensorrt::Input(input_shapes[0]);
  auto spec = torch_tensorrt::ts::CompileSpec({in});
  spec.enabled_precisions.insert(torch_tensorrt::DataType::kHalf);

  mod.to(torch::kHalf);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

TEST_P(CppAPITests, InputsUseDefaultFP16WithoutFP16Enabled) {
  torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = torch_tensorrt::Input(input_shapes[0]);
  auto spec = torch_tensorrt::ts::CompileSpec({in});

  mod.to(torch::kHalf);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

TEST_P(CppAPITests, InputsRespectUserSettingFP16WeightsFP32In) {
  torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto in = torch_tensorrt::Input(input_shapes[0]);
  in.dtype = torch::kFloat;
  auto spec = torch_tensorrt::ts::CompileSpec({in});
  spec.enabled_precisions.insert(torch_tensorrt::DataType::kHalf);

  mod.to(torch::kHalf);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

TEST_P(CppAPITests, InputsRespectUserSettingFP32WeightsFP16In) {
  torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = torch_tensorrt::Input(input_shapes[0]);
  in.dtype = torch::kF16;
  auto spec = torch_tensorrt::ts::CompileSpec({in});
  spec.enabled_precisions.insert(torch_tensorrt::DataType::kHalf);

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        PathAndInput({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, {at::kFloat} /*unused*/, 2e-5})));
