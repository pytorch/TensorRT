#include <string>
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "tests/util/util.h"
#include "core/compiler.h"

TEST(Converters, ATenUpsampleNearest1dConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2)
        %4 : None = prim::Constant()
        %5 : Tensor = aten::upsample_nearest1d(%0, %3, %4)
        return (%5))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 3D for TensorRT upsample_nearest1d
    auto in = at::randint(1, 10, {10, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest2dConvertsCorrectly2dOutputSize) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2)
        %4 : None = prim::Constant()
        %5 : Tensor = aten::upsample_nearest2d(%0, %3, %4, %4)
        return (%5))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 4D for TensorRT upsample_nearest2d
    auto in = at::randint(1, 10, {10, 2, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest3dConvertsCorrectly3dOutputSize) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2, %2)
        %4 : None = prim::Constant()
        %5 : Tensor = aten::upsample_nearest3d(%0, %3, %4, %4, %4)
        return (%5))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 5D for TensorRT upsample_nearest3d
    auto in = at::randint(1, 10, {10, 2, 2, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleLinear1dConvertsCorrectly) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2)
        %4 : bool = prim::Constant[value=1]()
        %5 : None = prim::Constant()
        %6 : Tensor = aten::upsample_linear1d(%0, %3, %4, %5)
        return (%6))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 3D for TensorRT upsample_linear1d
    auto in = at::randint(1, 10, {10, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleBilinear2dConvertsCorrectly2dOutputSize) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2)
        %4 : bool = prim::Constant[value=1]()
        %5 : None = prim::Constant()
        %6 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %5, %5)
        return (%6))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 4D for TensorRT upsample_bilinear2d
    auto in = at::randint(1, 10, {10, 2, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleTrilinear3dConvertsCorrectly3dOutputSize) {
    const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2, %2)
        %4 : bool = prim::Constant[value=1]()
        %5 : None = prim::Constant()
        %6 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %5, %5, %5)
        return (%6))IR";
    
    auto g = std::make_shared<torch::jit::Graph>();

    torch::jit::parseIR(graph, &*g);

    // Input Tensor needs to be 5D for TensorRT upsample_trilinear3d
    auto in = at::randint(1, 10, {10, 2, 2, 2, 2}, {at::kCUDA});

    auto jit_in = at::clone(in);
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});

    auto trt_in = at::clone(in);
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});

    auto trt = trt_results[0].reshape(jit_results[0].sizes());

    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}