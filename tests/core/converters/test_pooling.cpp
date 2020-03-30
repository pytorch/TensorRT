#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

#define LAYER_TEST //used to set threshold for diff

TEST(Converters, ATenMaxPool2DConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %5 : bool = prim::Constant[value=0]()
        %6 : int[] = prim::ListConstruct(%1, %1)
        %7 : int[] = prim::ListConstruct(%2, %2)
        %8 : int[] = prim::ListConstruct(%3, %3)
        %10 : Tensor = aten::max_pool2d(%0, %8, %7, %6, %7, %5)
        return (%10))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    //PyTorch MaxPool needs a 3D input
    auto in = at::randint(-5, 5, {1, 4, 4}, at::kCUDA);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAdaptiveAvgPool2DConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=3]()
        %3 : int = prim::Constant[value=4]()
        %6 : int[] = prim::ListConstruct(%2, %3)
        %10 : Tensor = aten::adaptive_avg_pool2d(%0, %6)
        return (%10))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    //PyTorch MaxPool needs a 3D input
    auto in = at::randint(-5, 5, {1, 12, 16}, at::kCUDA);

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}
