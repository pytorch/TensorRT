#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenSelectIntTwiceConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=3]()
        %4 : Tensor = aten::select(%0, %2, %2)
        %5 : Tensor = aten::select(%4, %2, %3)
        return (%5))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    auto in = at::randint(1, 10, {4, 4, 4}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}