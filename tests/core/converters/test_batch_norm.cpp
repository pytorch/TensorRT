#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenBatchNormConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor,
            %1: Float(5:1),
            %2: Float(5:1),
            %3: Float(5:1),
            %4: Float(5:1)):
        %5 : bool = prim::Constant[value=0]()
        %6 : float = prim::Constant[value=1.0000000000000001e-05]()
        %7 : float = prim::Constant[value=0.10000000000000001]()
        %8 : Tensor = aten::batch_norm(%0, %1, %2, %3, %4, %5, %6, %7, %5)
        return (%8))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});
    auto gamma = at::randint(1, 10, {5}, {at::kCUDA});
    auto beta = at::randint(1, 10, {5}, {at::kCUDA});
    auto mean = at::randint(1, 10, {5}, {at::kCUDA});
    auto var = at::randint(1, 10, {5}, {at::kCUDA});

    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {gamma, beta, mean, var});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    params = trtorch::core::conversion::get_named_params(g->inputs(), {gamma, beta, mean, var});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
