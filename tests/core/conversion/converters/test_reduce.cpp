#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

namespace {
std::string gen_basic_graph(const std::string& op) {
  return R"IR(
    graph(%0 : Tensor):
      %4 : None = prim::Constant()
      %5 : Tensor = aten::)IR" +
      op + R"IR((%0, %4)
      return (%5))IR";
}

std::string gen_min_max_graph(const std::string& op) {
  return R"IR(
    graph(%0 : Tensor):
      %5 : Tensor = aten::)IR" +
      op + R"IR((%0)
      return (%5))IR";
}

std::string gen_dim_graph(const std::string& op) {
  return R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int[] = prim::ListConstruct(%1)
        %3 : bool = prim::Constant[value=0]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::)IR" +
      op + R"IR((%0, %2, %3, %4)
        return (%5))IR";
}

std::string gen_multidim_graph(const std::string& op) {
  return R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=0]()
      %2 : int = prim::Constant[value=1]()
      %3 : int[] = prim::ListConstruct(%1, %2)
      %4 : bool = prim::Constant[value=0]()
      %5 : None = prim::Constant()
      %6 : Tensor = aten::)IR" +
      op + R"IR((%0, %3, %4, %5)
      return (%6))IR";
}

std::string gen_keepdim_graph(const std::string& op) {
  return R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int[] = prim::ListConstruct(%1)
        %3 : bool = prim::Constant[value=1]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::)IR" +
      op + R"IR((%0, %2, %3, %4)
        return (%5))IR";
}

void test_body(const std::string& graph, at::Tensor& in) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}
} // namespace

#define converts_correctly(op, name)                 \
  TEST(Converters, ATen##name##ConvertsCorrectly) {  \
    const auto graph = gen_basic_graph(#op);         \
    auto in = at::randint(-5, 5, {4, 4}, at::kCUDA); \
    test_body(graph, in);                            \
  }

converts_correctly(sum, Sum);
converts_correctly(prod, Prod);
converts_correctly(mean, Mean);

#undef converts_correctly

#define min_max_converts_correctly(op, name)         \
  TEST(Converters, ATen##name##ConvertsCorrectly) {  \
    const auto graph = gen_min_max_graph(#op);       \
    auto in = at::randint(-5, 5, {4, 4}, at::kCUDA); \
    test_body(graph, in);                            \
  }

min_max_converts_correctly(max, Max);
min_max_converts_correctly(min, Min);

#undef min_max_converts_correctly

#define converts_dim_correctly(op, name)                \
  TEST(Converters, ATen##name##DimConvertsCorrectly) {  \
    const auto graph = gen_dim_graph(#op);              \
    auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA); \
    test_body(graph, in);                               \
  }

converts_dim_correctly(sum, Sum);
converts_dim_correctly(mean, Mean);

#undef converts_dim_correctly

#define converts_multidims_correctly(op, name)               \
  TEST(Converters, ATen##name##MultiDimsConvertsCorrectly) { \
    const auto graph = gen_multidim_graph(#op);              \
    auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);      \
    test_body(graph, in);                                    \
  }

converts_multidims_correctly(sum, Sum);
converts_multidims_correctly(mean, Mean);

#undef converts_multidims_correctly

#define converts_keepdims_correctly(op, name)               \
  TEST(Converters, ATen##name##KeepDimsConvertsCorrectly) { \
    const auto graph = gen_keepdim_graph(#op);              \
    auto in = at::randint(-5, 5, {4, 4}, at::kCUDA);        \
    test_body(graph, in);                                   \
  }

converts_keepdims_correctly(sum, Sum);
converts_keepdims_correctly(mean, Mean);

#undef converts_keepdims_correctly

TEST(Converters, ATenSumDimNegOneIndexConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=0]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenSumDimNegOneIndexKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=1]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenSumDimNegIndexConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-2]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=0]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenSumDimNegIndexKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-2]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=1]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenProdDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=0]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::prod(%0, %1, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenProdKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %3 : bool = prim::Constant[value=1]()
        %4 : None = prim::Constant()
        %5 : Tensor = aten::prod(%0, %1, %3, %4)
        return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4}, at::kCUDA);
  test_body(graph, in);
}