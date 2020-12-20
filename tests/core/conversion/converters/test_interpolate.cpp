#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenUpsampleNearest1dOutputSizeConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest1dScaleFactorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=8]()
        %2 : int[] = prim::ListConstruct(%1)
        %3 : float = prim::Constant[value=4.0]()
        %5 : Tensor = aten::upsample_nearest1d(%0, %2, %3)
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest2dOutputSizeConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest2dScaleFactorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=8]()
        %3 : int[] = prim::ListConstruct(%2, %2)
        %4 : float = prim::Constant[value=4.0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest3dOutputSizeConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleNearest3dScaleFactorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=8]()
        %3 : int[] = prim::ListConstruct(%2, %2, %2)
        %4 : float = prim::Constant[value=4.0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleLinear1dOutputSizeWithAlignCornersConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleLinear1dOutputSizeWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2)
        %4 : bool = prim::Constant[value=0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleLinear1dScaleFactorWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=8]()
        %3 : int[] = prim::ListConstruct(%2)
        %4 : bool = prim::Constant[value=0]()
        %5 : float = prim::Constant[value=4.0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleBilinear2dOutputSizeWithAlignCornersConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleBilinear2dOutputSizeWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2)
        %4 : bool = prim::Constant[value=0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleBilinear2dScaleFactorWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=8]()
        %3 : int[] = prim::ListConstruct(%2, %2)
        %4 : bool = prim::Constant[value=0]()
        %5 : float = prim::Constant[value=4.0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleTrilinear3dOutputSizeWithAlignCornersConvertsCorrectly) {
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleTrilinear3dOutputSizeWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=10]()
        %3 : int[] = prim::ListConstruct(%2, %2, %2)
        %4 : bool = prim::Constant[value=0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenUpsampleTrilinear3dScaleFactorWithoutAlignCornersConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=8]()
        %3 : int[] = prim::ListConstruct(%2, %2, %2)
        %4 : bool = prim::Constant[value=0]()
        %5 : float = prim::Constant[value=4.0]()
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

  trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in});
  trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}