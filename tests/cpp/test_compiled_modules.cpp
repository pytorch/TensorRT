#include "cpp_api_test.h"

TEST_P(CppAPITests, CompiledModuleIsClose) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  std::vector<torch_tensorrt::Input> shapes;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randn(input_shapes[i], {at::kCUDA}).to(input_types[i]);
    if (input_types[i] == at::kInt || input_types[i] == at::kLong) {
      auto in = at::randint(0, 2, input_shapes[i], {at::kCUDA}).to(input_types[i]);
    }

    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
    auto in_spec = torch_tensorrt::Input(input_shapes[i]);
    in_spec.dtype = input_types[i];
    shapes.push_back(in_spec);
    std::cout << in_spec << std::endl;
  }

  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  if (jit_results_ivalues.isTuple()) {
    auto tuple = jit_results_ivalues.toTuple();
    for (auto t : tuple->elements()) {
      jit_results.push_back(t.toTensor());
    }
  } else {
    jit_results.push_back(jit_results_ivalues.toTensor());
  }

  auto spec = torch_tensorrt::ts::CompileSpec(shapes);
  spec.truncate_long_and_double = true;

  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  if (trt_results_ivalues.isTuple()) {
    auto tuple = trt_results_ivalues.toTuple();
    for (auto t : tuple->elements()) {
      trt_results.push_back(t.toTensor());
    }
  } else {
    trt_results.push_back(trt_results_ivalues.toTensor());
  }

  for (size_t i = 0; i < trt_results.size(); i++) {
    ASSERT_TRUE(torch_tensorrt::tests::util::cosineSimEqual(jit_results[i], trt_results[i].reshape_as(jit_results[i])));
  }
}

#ifdef DISABLE_TEST_IN_CI

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CppAPITests);

#else

INSTANTIATE_TEST_SUITE_P(
    DISABLED_CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        PathAndInput({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/efficientnet_b0_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}}),
        PathAndInput({"tests/modules/vit_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}})));

#endif
