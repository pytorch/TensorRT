#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

#define ATEN_INTERPOLATE_TESTS(name, graph_src, input_shape)                             \
  TEST(Converters, name##StaticConvertsCorrectly) {                                      \
    const auto graph = graph_src;                                                        \
                                                                                         \
    auto g = std::make_shared<torch::jit::Graph>();                                      \
    torch::jit::parseIR(graph, &*g);                                                     \
                                                                                         \
    auto in = at::randint(1, 10, input_shape, {at::kCUDA});                              \
    auto jit_in = at::clone(in);                                                         \
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});          \
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});              \
                                                                                         \
    auto trt_in = at::clone(in);                                                         \
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});               \
                                                                                         \
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in});        \
    auto trt = trt_results[0].reshape(jit_results[0].sizes());                           \
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));           \
  }                                                                                      \
                                                                                         \
  TEST(Converters, name##DynamicConvertsCorrectly) {                                     \
    const auto graph = graph_src;                                                        \
                                                                                         \
    auto g = std::make_shared<torch::jit::Graph>();                                      \
    torch::jit::parseIR(graph, &*g);                                                     \
                                                                                         \
    auto in = at::randint(1, 10, input_shape, {at::kCUDA});                              \
    auto jit_in = at::clone(in);                                                         \
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});          \
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});              \
                                                                                         \
    auto trt_in = at::clone(in);                                                         \
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});               \
                                                                                         \
    auto trt_results = trtorch::tests::util::RunGraphEngineDynamic(g, params, {trt_in}); \
    auto trt = trt_results[0].reshape(jit_results[0].sizes());                           \
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));           \
  }

#define ATEN_INTERPOLATE_STATIC_ONLY_TEST(name, graph_src, input_shape)           \
  TEST(Converters, name##StaticConvertsCorrectly) {                               \
    const auto graph = graph_src;                                                 \
                                                                                  \
    auto g = std::make_shared<torch::jit::Graph>();                               \
    torch::jit::parseIR(graph, &*g);                                              \
                                                                                  \
    auto in = at::randint(1, 10, input_shape, {at::kCUDA});                       \
    auto jit_in = at::clone(in);                                                  \
    auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});   \
    auto jit_results = trtorch::tests::util::RunGraph(g, params, {jit_in});       \
                                                                                  \
    auto trt_in = at::clone(in);                                                  \
    params = trtorch::core::conversion::get_named_params(g->inputs(), {});        \
                                                                                  \
    auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {trt_in}); \
    auto trt = trt_results[0].reshape(jit_results[0].sizes());                    \
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt, 2e-6));    \
  }

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest1dOutputSize,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2)
      %4 : None = prim::Constant()
      %5 : Tensor = aten::upsample_nearest1d(%0, %3, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest1dScales,
    R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=8]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : float = prim::Constant[value=4.0]()
      %5 : Tensor = aten::upsample_nearest1d(%0, %2, %3)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest1dVecScaleFactors,
    R"IR(
    graph(%0 : Tensor):
      %2 : None = prim::Constant()
      %3 : float = prim::Constant[value=4.0]()
      %4 : float[] = prim::ListConstruct(%3)
      %5 : Tensor = aten::upsample_nearest1d(%0, %2, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest2dOutputSize,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : None = prim::Constant()
      %5 : Tensor = aten::upsample_nearest2d(%0, %3, %4, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest2dScales,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : float = prim::Constant[value=4.0]()
      %5 : Tensor = aten::upsample_nearest2d(%0, %3, %4, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest2dVecScaleFactors,
    R"IR(
    graph(%0 : Tensor):
      %2 : None = prim::Constant()
      %3 : float = prim::Constant[value=4.0]()
      %4 : float[] = prim::ListConstruct(%3, %3)
      %5 : Tensor = aten::upsample_nearest2d(%0, %2, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest3dOutputSize,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : None = prim::Constant()
      %5 : Tensor = aten::upsample_nearest3d(%0, %3, %4, %4, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest3dScales,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : float = prim::Constant[value=4.0]()
      %5 : Tensor = aten::upsample_nearest3d(%0, %3, %4, %4, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleNearest3dVecScaleFactors,
    R"IR(
    graph(%0 : Tensor):
      %2 : None = prim::Constant()
      %3 : float = prim::Constant[value=4.0]()
      %4 : float[] = prim::ListConstruct(%3, %3, %3)
      %5 : Tensor = aten::upsample_nearest3d(%0, %2, %4)
      return (%5))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleLinear1dOutputSizeWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2)
      %4 : bool = prim::Constant[value=1]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_linear1d(%0, %3, %4, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleLinear1dOutputSizeWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2)
      %4 : bool = prim::Constant[value=0]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_linear1d(%0, %3, %4, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleLinear1dScalesWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2)
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : Tensor = aten::upsample_linear1d(%0, %3, %4, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleLinear1dScalesWithAlignCorners,
    R"IR(
     graph(%0 : Tensor):
       %2 : int = prim::Constant[value=8]()
       %3 : int[] = prim::ListConstruct(%2)
       %4 : bool = prim::Constant[value=1]()
       %5 : float = prim::Constant[value=4.0]()
       %6 : Tensor = aten::upsample_linear1d(%0, %3, %4, %5)
       return (%6))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleLinear1dVecScaleFactorsWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %3 : None = prim::Constant()
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : float[] = prim::ListConstruct(%5)
      %7 : Tensor = aten::upsample_linear1d(%0, %3, %4, %6)
      return (%7))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleLinear1dVecScaleFactorsWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
     %3 : None = prim::Constant()
     %4 : bool = prim::Constant[value=1]()
     %5 : float = prim::Constant[value=4.0]()
     %6 : float[] = prim::ListConstruct(%5)
     %7 : Tensor = aten::upsample_linear1d(%0, %3, %4, %6)
     return (%7))IR",
    std::vector<int64_t>({10, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleBilinear2dOutputSizeWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : bool = prim::Constant[value=1]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleBilinear2dOutputSizeWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : bool = prim::Constant[value=0]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleBilinear2dScalesWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleBilinear2dScalesWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2)
      %4 : bool = prim::Constant[value=1]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleBilinear2dVecScaleFactorsWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %3 : None = prim::Constant()
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : float[] = prim::ListConstruct(%5, %5)
      %7 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %6)
      return (%7))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleBilinear2dVecScaleFactorsWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %3 : None = prim::Constant()
      %4 : bool = prim::Constant[value=1]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : float[] = prim::ListConstruct(%5, %5)
      %7 : Tensor = aten::upsample_bilinear2d(%0, %3, %4, %6)
      return (%7))IR",
    std::vector<int64_t>({10, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleTrilinear3dOutputSizeWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : bool = prim::Constant[value=1]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %5, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleTrilinear3dOutputSizeWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=10]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : bool = prim::Constant[value=0]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %5, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleTrilinear3dScalesWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %5, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleTrilinear3dScalesWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %2 : int = prim::Constant[value=8]()
      %3 : int[] = prim::ListConstruct(%2, %2, %2)
      %4 : bool = prim::Constant[value=1]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %5, %5, %5)
      return (%6))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_TESTS(
    ATenUpsampleTrilinear3dVecScaleFactorsWithoutAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %3 : None = prim::Constant()
      %4 : bool = prim::Constant[value=0]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : float[] = prim::ListConstruct(%5, %5, %5)
      %7 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %6)
      return (%7))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));

ATEN_INTERPOLATE_STATIC_ONLY_TEST(
    ATenUpsampleTrilinear3dVecScaleFactorsWithAlignCorners,
    R"IR(
    graph(%0 : Tensor):
      %3 : None = prim::Constant()
      %4 : bool = prim::Constant[value=1]()
      %5 : float = prim::Constant[value=4.0]()
      %6 : float[] = prim::ListConstruct(%5, %5, %5)
      %7 : Tensor = aten::upsample_trilinear3d(%0, %3, %4, %6)
      return (%7))IR",
    std::vector<int64_t>({10, 2, 2, 2, 2}));