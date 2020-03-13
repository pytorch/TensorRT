#include "module_test.h"

TEST_P(ModuleTests, ModuleAsEngineIsClose) {
    std::vector<at::Tensor> inputs;
    std::vector<torch::jit::IValue> inputs_ivalues;
    for (auto in_shape : input_shapes) {
        inputs.push_back(at::randint(5, in_shape, {at::kCUDA}));
        inputs_ivalues.push_back(inputs[inputs.size() - 1].clone());
    }

    torch::jit::IValue jit_results_ivalues = trtorch::tests::util::RunModuleForward(mod, inputs_ivalues);
    std::vector<at::Tensor> jit_results;
    jit_results.push_back(jit_results_ivalues.toTensor());
    auto trt_results = trtorch::tests::util::RunModuleForwardAsEngine(mod, inputs);
    
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0])));
}

INSTANTIATE_TEST_SUITE_P(ModuleAsEngineForwardIsCloseSuite,
                         ModuleTests,
                         testing::Values(
                             PathAndInSize({"tests/models/lenet.jit.pt",
                                            {{1,1,28,28}}}),
                             PathAndInSize({"tests/models/resnet18.jit.pt",
                                            {{1,3,224,224}}}),
                             PathAndInSize({"tests/models/resnet50.jit.pt",
                                            {{1,3,224,224}}})));
