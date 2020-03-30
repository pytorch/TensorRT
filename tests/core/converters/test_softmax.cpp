#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

#define LAYER_TEST //used to set threshold for diff

TEST(Converters, ATenSoftmax1DConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : None = prim::Constant()
        %2 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::softmax(%0, %2, %1)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    auto in = at::randint(0, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
    auto trt = trt_results[0].reshape_as(jit_results[0]);

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlySub3DIndex) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : None = prim::Constant()
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::softmax(%0, %2, %1)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    auto in = at::randint(0, 5, {1,2,2,2,2}, {at::kCUDA});

    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
    auto trt = trt_results[0].reshape_as(jit_results[0]);

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSoftmaxNDConvertsCorrectlyAbove3DIndex) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : None = prim::Constant()
        %2 : int = prim::Constant[value=3]()
        %3 : Tensor = aten::softmax(%0, %2, %1)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    auto in = at::randint(0, 5, {1,2,2,2,2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape_as(jit_results[0]);

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}
