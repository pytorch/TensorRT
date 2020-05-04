#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenReLUConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::relu(%0)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenSigmoidConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::sigmoid(%0)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenTanhConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %3 : Tensor = aten::tanh(%0)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

//TODO: Seems like the IR parser is not handling negative numbers well, need to follow up with the PyTorch Team
// TEST(Converters, ATenHardTanhConvertsCorrectly) {
//     const auto graph = R"IR(
//       graph(%0 : Tensor):
//         %1 : float = prim::Constant[value=-1.0]()
//         %2 : float = prim::Constant[value=1.0]()
//         %3 : Tensor = aten::hardtanh(%0, %1, %2)
//         return (%3))IR";

//     auto g = std::make_shared<torch::jit::Graph>();
//     torch::jit::script::parseIR(graph, &*g);

//     auto in = at::randint(-5, 5, {5}, {at::kCUDA});
//     auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
//     auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

//     in = at::clone(in);
//     params = trtorch::core::conversion::get_named_params(g->inputs(), {});
//     auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

//     ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
// }

TEST(Converters, ATenHardTanhCustomRangeConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : float = prim::Constant[value=0.0]()
        %2 : float = prim::Constant[value=6.0]()
        %3 : Tensor = aten::hardtanh(%0, %1, %2)
        return (%3))IR";

    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(-5, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

    in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

