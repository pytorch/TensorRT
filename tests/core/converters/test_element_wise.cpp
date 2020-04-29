#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

void pointwise_test_helper(std::string graph_ir) {
    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(graph_ir, &*g);

    auto in0 = at::randint(1, 5, {5}, {at::kCUDA});
    auto in1 = at::randint(1, 5, {5}, {at::kCUDA});
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {in0, in1});

    in0 = at::clone(in0);
    in1 = at::clone(in1);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in0, in1});

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}



TEST(Converters, ATenAddConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::add(%0, %1, %2)
        return (%3))IR";
    pointwise_test_helper(graph);
}


// TEST(Converters, ATenAddWithScaleConvertsCorrectly) {
//     const auto graph = R"IR(
//       graph(%0 : Tensor, %1 : Tensor):
//         %2 : int = prim::Constant[value=2]()
//         %3 : Tensor = aten::add(%0, %1, %2)
//         return (%3))IR";
//     pointwise_test_helper(graph);
// }

TEST(Converters, ATenSubConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::sub(%0, %1, %2)
        return (%3))IR";
    pointwise_test_helper(graph);
}

// TEST(Converters, ATenSubWithScaleConvertsCorrectly) {
//     const auto graph = R"IR(
//       graph(%0 : Tensor, %1 : Tensor):
//         %2 : float = prim::Constant[value=0.5]()
//         %3 : Tensor = aten::add(%0, %1, %2)
//         return (%3))IR";
//     pointwise_test_helper(graph);
// }

TEST(Converters, ATenMulConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::mul(%0, %1)
        return (%2))IR";
    pointwise_test_helper(graph);
}

TEST(Converters, ATenDivConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::div(%0, %1)
        return (%2))IR";
    pointwise_test_helper(graph);
}
