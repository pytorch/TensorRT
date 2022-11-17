#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

#define ATEN_INTERPOLATE_TESTS(name, graph_src, input_shape)                                    \
  TEST(Converters, name##StaticConvertsCorrectly) {                                             \
    const auto graph = graph_src;                                                               \
                                                                                                \
    auto g = std::make_shared<torch::jit::Graph>();                                             \
    torch::jit::parseIR(graph, &*g);                                                            \
                                                                                                \
    auto in = at::randint(1, 10, input_shape, {at::kCUDA});                                     \
    auto jit_in = at::clone(in);                                                                \
    auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                 \
    auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});              \
                                                                                                \
    auto trt_in = at::clone(in);                                                                \
    params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                      \
                                                                                                \
    auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});        \
    auto trt = trt_results[0].reshape(jit_results[0].sizes());                                  \
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));           \
  }                                                                                             \
                                                                                                \
  TEST(Converters, name##DynamicConvertsCorrectly) {                                            \
    const auto graph = graph_src;                                                               \
                                                                                                \
    auto g = std::make_shared<torch::jit::Graph>();                                             \
    torch::jit::parseIR(graph, &*g);                                                            \
                                                                                                \
    auto in = at::randint(1, 10, input_shape, {at::kCUDA});                                     \
    auto jit_in = at::clone(in);                                                                \
    auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                 \
    auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});              \
                                                                                                \
    auto trt_in = at::clone(in);                                                                \
    params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});                      \
                                                                                                \
    auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}); \
    auto trt = trt_results[0].reshape(jit_results[0].sizes());                                  \
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));           \
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

TEST(Converters, ATenFrobeniusNorm) {
  const auto graph = R"IR(
      graph(%x : Tensor):
        %0 : int = prim::Constant[value=0]()
        %keep : bool = prim::Constant[value=0]()
        %dims : int[] = prim::ListConstruct(%0)
        %out : Tensor = aten::frobenius_norm(%x, %dims, %keep)
        return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto x = at::randn({5, 5, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenFrobeniusNormKeep) {
  const auto graph = R"IR(
      graph(%x : Tensor):
        %1 : int = prim::Constant[value=-1]()
        %keep : bool = prim::Constant[value=1]()
        %dims : int[] = prim::ListConstruct(%1)
        %out : Tensor = aten::frobenius_norm(%x, %dims, %keep)
        return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto x = at::randn({5, 5, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenFrobeniusNormMatrix) {
  const auto graph = R"IR(
      graph(%x : Tensor):
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=-1]()
        %keep : bool = prim::Constant[value=0]()
        %dims : int[] = prim::ListConstruct(%0, %1)
        %out : Tensor = aten::frobenius_norm(%x, %dims, %keep)
        return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto x = at::randn({3, 5, 7, 11, 13}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenLinAlgNorm_None) {
  const auto graph = R"IR(
    graph(%x : Tensor):
      %none : NoneType = prim::Constant()
      %keep : bool = prim::Constant[value=0]()
      %out : Tensor = aten::linalg_norm(%x, %none, %none, %keep, %none)
      return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());
  auto x = at::randn({5, 5, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenLinAlgNorm_1D) {
  const auto graph = R"IR(
    graph(%x : Tensor):
      %1 : int = prim::Constant[value=1]()
      %none : NoneType = prim::Constant()
      %keep : bool = prim::Constant[value=0]()
      %dims : int[] = prim::ListConstruct(%1)
      %out : Tensor = aten::linalg_norm(%x, %none, %dims, %keep, %none)
      return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto x = at::randn({5, 5, 5}, {at::kCUDA});
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}

TEST(Converters, ATenLinAlgNorm_2D) {
  const auto graph = R"IR(
    graph(%x : Tensor):
      %0 : int = prim::Constant[value=0]()
      %1 : int = prim::Constant[value=-1]()
      %none : NoneType = prim::Constant()
      %keep : bool = prim::Constant[value=1]()
      %dims : int[] = prim::ListConstruct(%0, %1)
      %float : int = prim::Constant[value=6]()
      %out : Tensor = aten::linalg_norm(%x, %none, %dims, %keep, %float)
      return (%out))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto x = at::randn({5, 5, 5}, {at::kCUDA}).to(at::kHalf);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {x});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {x});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
}
