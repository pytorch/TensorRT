#include "module_test.h"

TEST_P(ModuleTests, SerializedModuleIsStillCorrect) {
    std::vector<torch::jit::IValue> post_serialized_inputs_ivalues;
    std::vector<torch::jit::IValue> pre_serialized_inputs_ivalues;
    for (auto in_shape : input_shapes) {
        auto in = at::randint(5, in_shape, {at::kCUDA});
        post_serialized_inputs_ivalues.push_back(in.clone());
        pre_serialized_inputs_ivalues.push_back(in.clone());
    }

    auto pre_serialized_mod = trtorch::CompileGraph(mod, input_shapes);
    torch::jit::IValue pre_serialized_results_ivalues = trtorch::tests::util::RunModuleForward(pre_serialized_mod, pre_serialized_inputs_ivalues);
    std::vector<at::Tensor> pre_serialized_results;
    pre_serialized_results.push_back(pre_serialized_results_ivalues.toTensor());

    pre_serialized_mod.save("test_serialization_mod.ts");
    auto post_serialized_mod = torch::jit::load("test_serialization_mod.ts");

    torch::jit::IValue post_serialized_results_ivalues = trtorch::tests::util::RunModuleForward(post_serialized_mod, post_serialized_inputs_ivalues);
    std::vector<at::Tensor> post_serialized_results;
    post_serialized_results.push_back(post_serialized_results_ivalues.toTensor());

    for (size_t i = 0; i < pre_serialized_results.size(); i++) {
        ASSERT_TRUE(trtorch::tests::util::almostEqual(post_serialized_results[i], pre_serialized_results[i].reshape_as(post_serialized_results[i]), 2e-5));
    }
}


INSTANTIATE_TEST_SUITE_P(CompiledModuleForwardIsCloseSuite,
                         ModuleTests,
                         testing::Values(
                            PathAndInSize({"tests/modules/resnet18_traced.jit.pt",
                                          {{1,3,224,224}}}),
                            PathAndInSize({"tests/modules/interpolate_traced.jit.pt",
                                          {{1,3,5,5,5}}})));