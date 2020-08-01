#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenLoopConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : Tensor, %4 : Tensor, %5 : Tensor, %8 : Tensor):
        %22 : int = prim::Constant[value=1]()
        %10 : bool = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %98 : Tensor = aten::tanh(%1)
        %7 : int = aten::size(%0, %6)
        %99 : Tensor, %95 : Tensor = prim::Loop(%7, %10, %98, %1)
          block0(%90 : int, %96 : Tensor, %93 : Tensor):
            %16 : Tensor = aten::select(%0, %6, %90)
            %18 : Tensor = aten::matmul(%16, %2)
            %21 : Tensor = aten::matmul(%93, %3)
            %23 : Tensor = aten::add(%18, %21, %22)
            %26 : Tensor = aten::add(%23, %4, %22)
            %94 : Tensor = aten::tanh(%26)
            %31 : Tensor = aten::matmul(%94, %5)
            %34 : Tensor = aten::add(%31, %8, %22)
            %97 : Tensor = aten::tanh(%34)
            -> (%10, %97, %94)
        return (%99))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    auto x = at::randn({5, 5, 3}, {at::kCUDA});
    auto h = at::randn({5, 5}, {at::kCUDA});
    auto Wh = at::randn({3, 5}, {at::kCUDA});
    auto Uh = at::randn({5, 5}, {at::kCUDA});
    auto bh = at::randn({5, 5}, {at::kCUDA});
    auto Wy = at::randn({5, 5}, {at::kCUDA});
    auto by = at::randn({5, 5}, {at::kCUDA});

    auto jit_x = at::clone(x);
    auto jit_h = at::clone(h);
    auto jit_Wh = at::clone(Wh);
    auto jit_Uh = at::clone(Uh);
    auto jit_bh = at::clone(bh);
    auto jit_Wy = at::clone(Wy);
    auto jit_by = at::clone(by);
    
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_x, jit_h, jit_Wh, jit_Uh, jit_bh, jit_Wy, jit_by});

    auto trt_x = at::clone(x);
    auto trt_h = at::clone(h);
    auto trt_Wh = at::clone(Wh);
    auto trt_Uh = at::clone(Uh);
    auto trt_bh = at::clone(bh);
    auto trt_Wy = at::clone(Wy);
    auto trt_by = at::clone(by);

    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_x, trt_h, trt_Wh, trt_Uh, trt_bh, trt_Wy, trt_by});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}