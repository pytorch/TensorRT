#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenReLUConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::relu(%0)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::script::parseIR(graph, &*g);
    
    auto in = at::randint(-5, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0]));
}
