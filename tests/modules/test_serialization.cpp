#include "module_test.h"

std::vector<trtorch::CompileSpec::InputRange> toInputRangesDynamic(std::vector<std::vector<int64_t>> opts) {
  std::vector<trtorch::CompileSpec::InputRange> a;

  for (auto opt : opts) {
    std::vector<int64_t> min_range(opt);
    std::vector<int64_t> max_range(opt);

    min_range[3] = ceil(opt[3] / 2.0);
    max_range[3] = 2 * opt[3];
    min_range[2] = ceil(opt[2] / 2.0);
    max_range[2] = 2 * opt[2];

    a.push_back(trtorch::CompileSpec::InputRange(min_range, opt, max_range));
  }

  return std::move(a);
}

TEST_P(ModuleTests, SerializedModuleIsStillCorrect) {
  std::vector<torch::jit::IValue> post_serialized_inputs_ivalues;
  std::vector<torch::jit::IValue> pre_serialized_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    post_serialized_inputs_ivalues.push_back(in.clone());
    pre_serialized_inputs_ivalues.push_back(in.clone());
  }

  auto pre_serialized_mod = trtorch::CompileGraph(mod, input_shapes);
  torch::jit::IValue pre_serialized_results_ivalues =
      trtorch::tests::util::RunModuleForward(pre_serialized_mod, pre_serialized_inputs_ivalues);
  std::vector<at::Tensor> pre_serialized_results;
  pre_serialized_results.push_back(pre_serialized_results_ivalues.toTensor());

  pre_serialized_mod.save("test_serialization_mod.ts");
  auto post_serialized_mod = torch::jit::load("test_serialization_mod.ts");

  torch::jit::IValue post_serialized_results_ivalues =
      trtorch::tests::util::RunModuleForward(post_serialized_mod, post_serialized_inputs_ivalues);
  std::vector<at::Tensor> post_serialized_results;
  post_serialized_results.push_back(post_serialized_results_ivalues.toTensor());

  for (size_t i = 0; i < pre_serialized_results.size(); i++) {
    ASSERT_TRUE(trtorch::tests::util::almostEqual(
        post_serialized_results[i], pre_serialized_results[i].reshape_as(post_serialized_results[i]), 2e-5));
  }
}

TEST_P(ModuleTests, SerializedDynamicModuleIsStillCorrect) {
  std::vector<torch::jit::IValue> post_serialized_inputs_ivalues;
  std::vector<torch::jit::IValue> pre_serialized_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    post_serialized_inputs_ivalues.push_back(in.clone());
    pre_serialized_inputs_ivalues.push_back(in.clone());
  }

  auto pre_serialized_mod = trtorch::CompileGraph(mod, toInputRangesDynamic(input_shapes));
  torch::jit::IValue pre_serialized_results_ivalues =
      trtorch::tests::util::RunModuleForward(pre_serialized_mod, pre_serialized_inputs_ivalues);
  std::vector<at::Tensor> pre_serialized_results;
  pre_serialized_results.push_back(pre_serialized_results_ivalues.toTensor());

  pre_serialized_mod.save("test_serialization_mod.ts");
  auto post_serialized_mod = torch::jit::load("test_serialization_mod.ts");

  torch::jit::IValue post_serialized_results_ivalues =
      trtorch::tests::util::RunModuleForward(post_serialized_mod, post_serialized_inputs_ivalues);
  std::vector<at::Tensor> post_serialized_results;
  post_serialized_results.push_back(post_serialized_results_ivalues.toTensor());

  for (size_t i = 0; i < pre_serialized_results.size(); i++) {
    ASSERT_TRUE(trtorch::tests::util::almostEqual(
        post_serialized_results[i], pre_serialized_results[i].reshape_as(post_serialized_results[i]), 2e-5));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    ModuleTests,
    testing::Values(
        PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/pooling_traced.jit.pt", {{1, 3, 10, 10}}})));