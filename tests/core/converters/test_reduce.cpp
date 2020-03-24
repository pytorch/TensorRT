#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenMeanConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int[] = prim::ListConstruct(%1)    
        %3 : bool = prim::Constant[value=0]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::mean(%0, %2, %3, %4) 
        return (%5))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {4, 4}, at::kCUDA);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
    
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenMeanKeepDimsConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int[] = prim::ListConstruct(%1)    
        %3 : bool = prim::Constant[value=1]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::mean(%0, %2, %3, %4) 
        return (%5))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {4, 4}, at::kCUDA);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
    
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0]));
}