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

ATEN_INTERPOLATE_TESTS(
    ATenNormOrder1RemoveDims,
    R"IR(
      graph(%x.1 : Tensor):
              %2 : int[] = prim::Constant[value=[1, 2]]()
              %3 : int = prim::Constant[value=1]()
              %4 : bool = prim::Constant[value=0]()
              %5 : Tensor = aten::norm(%x.1, %3, %2, %4)
              return (%5))IR",
    std::vector<int64_t>({3, 4, 3}));

ATEN_INTERPOLATE_TESTS(
    ATenNormOrder2RemoveDims,
    R"IR(
      graph(%x.1 : Tensor):
              %2 : int[] = prim::Constant[value=[1, 2]]()
              %3 : int = prim::Constant[value=2]()
              %4 : bool = prim::Constant[value=0]()
              %5 : Tensor = aten::norm(%x.1, %3, %2, %4)
              return (%5))IR",
    std::vector<int64_t>({3, 4, 3}));

ATEN_INTERPOLATE_TESTS(
    ATenNormOrder2KeepDims,
    R"IR(
      graph(%x.1 : Tensor):
              %2 : int[] = prim::Constant[value=[1]]()
              %3 : int = prim::Constant[value=2]()
              %4 : bool = prim::Constant[value=1]()
              %5 : Tensor = aten::norm(%x.1, %3, %2, %4)
              return (%5))IR",
    std::vector<int64_t>({3, 4, 3}));
