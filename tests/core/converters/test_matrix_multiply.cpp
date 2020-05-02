#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

// TODO: IR Parser doesnt work well with neg numbers
TEST(Converters, ATenMMConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::matmul(%0, %1)
        return (%2))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in1 = at::randint(0, 5, {2, 3}, {at::kCUDA});
    auto in2 = at::randint(0, 5, {3, 3}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in1, in2});

    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in1, in2});
    auto trt = trt_results[0].reshape_as(jit_results[0]);

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}
