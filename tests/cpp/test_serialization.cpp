#include "cpp_api_test.h"

std::vector<torch_tensorrt::Input> toInputRangesDynamic(std::vector<std::vector<int64_t>> opts) {
  std::vector<torch_tensorrt::Input> a;

  for (auto opt : opts) {
    std::vector<int64_t> min_range(opt);
    std::vector<int64_t> max_range(opt);

    min_range[3] = ceil(opt[3] / 2.0);
    max_range[3] = 2 * opt[3];
    min_range[2] = ceil(opt[2] / 2.0);
    max_range[2] = 2 * opt[2];

    a.push_back(torch_tensorrt::Input(min_range, opt, max_range));
  }

  return std::move(a);
}

TEST_P(CppAPITests, SerializedModuleIsStillCorrect) {
  std::vector<torch::jit::IValue> post_serialized_inputs_ivalues;
  std::vector<torch::jit::IValue> pre_serialized_inputs_ivalues;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randint(5, input_shapes[i], {at::kCUDA}).to(input_types[i]);
    post_serialized_inputs_ivalues.push_back(in.clone());
    pre_serialized_inputs_ivalues.push_back(in.clone());
  }

  auto pre_serialized_mod = torch_tensorrt::ts::compile(mod, input_shapes);
  torch::jit::IValue pre_serialized_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(pre_serialized_mod, pre_serialized_inputs_ivalues);
  std::vector<at::Tensor> pre_serialized_results;
  pre_serialized_results.push_back(pre_serialized_results_ivalues.toTensor());

  pre_serialized_mod.save("test_serialization_mod.ts");
  auto post_serialized_mod = torch::jit::load("test_serialization_mod.ts");

  torch::jit::IValue post_serialized_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(post_serialized_mod, post_serialized_inputs_ivalues);
  std::vector<at::Tensor> post_serialized_results;
  post_serialized_results.push_back(post_serialized_results_ivalues.toTensor());

  for (size_t i = 0; i < pre_serialized_results.size(); i++) {
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
        post_serialized_results[i], pre_serialized_results[i].reshape_as(post_serialized_results[i]), threshold));
  }
}

TEST_P(CppAPITests, SerializedDynamicModuleIsStillCorrect) {
  std::vector<torch::jit::IValue> post_serialized_inputs_ivalues;
  std::vector<torch::jit::IValue> pre_serialized_inputs_ivalues;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randint(5, input_shapes[i], {at::kCUDA}).to(input_types[i]);
    post_serialized_inputs_ivalues.push_back(in.clone());
    pre_serialized_inputs_ivalues.push_back(in.clone());
  }

  auto pre_serialized_mod =
      torch_tensorrt::ts::compile(mod, torch_tensorrt::ts::CompileSpec(toInputRangesDynamic(input_shapes)));
  torch::jit::IValue pre_serialized_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(pre_serialized_mod, pre_serialized_inputs_ivalues);
  std::vector<at::Tensor> pre_serialized_results;
  pre_serialized_results.push_back(pre_serialized_results_ivalues.toTensor());

  pre_serialized_mod.save("test_serialization_mod.ts");
  auto post_serialized_mod = torch::jit::load("test_serialization_mod.ts");

  torch::jit::IValue post_serialized_results_ivalues =
      torch_tensorrt::tests::util::RunModuleForward(post_serialized_mod, post_serialized_inputs_ivalues);
  std::vector<at::Tensor> post_serialized_results;
  post_serialized_results.push_back(post_serialized_results_ivalues.toTensor());

  for (size_t i = 0; i < pre_serialized_results.size(); i++) {
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
        post_serialized_results[i], pre_serialized_results[i].reshape_as(post_serialized_results[i]), threshold));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        PathAndInput({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5}),
        PathAndInput({"tests/modules/pooling_traced.jit.pt", {{1, 3, 10, 10}}, {at::kFloat}, 2e-5})));
