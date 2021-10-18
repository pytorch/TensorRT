#include "cpp_api_test.h"
#include "trtorch/logging.h"

TEST_P(CppAPITests, InputsUseDefaultFP32) {
  trtorch::logging::set_reportable_log_level(trtorch::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto in = trtorch::CompileSpec::Input(input_shapes[0]);
  auto spec = trtorch::CompileSpec({in});
  spec.enabled_precisions.insert(trtorch::CompileSpec::DataType::kHalf);

  auto trt_mod = trtorch::CompileGraph(mod, spec);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP32
}

TEST_P(CppAPITests, InputsUseDefaultFP16) {
  trtorch::logging::set_reportable_log_level(trtorch::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
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

TEST_P(CppAPITests, InputsUseDefaultFP16WithoutFP16Enabled) {
  trtorch::logging::set_reportable_log_level(trtorch::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = trtorch::CompileSpec::Input(input_shapes[0]);
  auto spec = trtorch::CompileSpec({in});

  mod.to(torch::kHalf);

  auto trt_mod = trtorch::CompileGraph(mod, spec);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

TEST_P(CppAPITests, InputsRespectUserSettingFP16WeightsFP32In) {
  trtorch::logging::set_reportable_log_level(trtorch::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto in = trtorch::CompileSpec::Input(input_shapes[0]);
  in.dtype = torch::kF32;
  auto spec = trtorch::CompileSpec({in});
  spec.enabled_precisions.insert(trtorch::CompileSpec::DataType::kHalf);

  mod.to(torch::kHalf);

  auto trt_mod = trtorch::CompileGraph(mod, spec);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
  // If exits without error successfully defaults to FP16
}

TEST_P(CppAPITests, InputsRespectUserSettingFP32WeightsFP16In) {
  trtorch::logging::set_reportable_log_level(trtorch::logging::Level::kINFO);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randn(in_shape, {at::kCUDA});
    trt_inputs_ivalues.push_back(in.clone().to(torch::kHalf));
  }

  auto in = trtorch::CompileSpec::Input(input_shapes[0]);
  in.dtype = torch::kF16;
  auto spec = trtorch::CompileSpec({in});
  spec.enabled_precisions.insert(trtorch::CompileSpec::DataType::kHalf);

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
