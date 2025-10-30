#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/torch.h"

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

void test_body(const std::string& graph, at::Tensor& in, bool dynamic = false) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  std::vector<at::Tensor> trt_results;
  if (dynamic) {
    trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in});
  } else {
    trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  }
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0]));
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

TEST(Converters, ATenSumBoolConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %4)
      return (%5))IR";
  auto in = at::randint(-1, 2, {4, 4, 4}, at::kCUDA).to(at::kBool);
  test_body(graph, in);
}

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

TEST(Converters, ATenSumDimNegOneIndexKeepDimsBoolTensorConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=1]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::sum(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(0, 2, {4, 4, 4}, at::kCUDA).to(torch::kBool);
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

TEST(Converters, ATenMaxKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x : Tensor):
          %2 : int = prim::Constant[value=-1]()
          %3 : bool = prim::Constant[value=1]()
          %keep.1 : Tensor, %6 : Tensor = aten::max(%x, %2, %3)
          return (%keep.1, %6))IR";

  auto in = at::randint(-5, 5, {4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenMeanDimNegOneIndexConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=0]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::mean(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenMeanDimNegOneIndexKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=1]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::mean(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenMeanDimNegIndexConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-2]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=0]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::mean(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenMeanDimNegIndexKeepDimsConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-2]()
      %2 : int[] = prim::ListConstruct(%1)
      %3 : bool = prim::Constant[value=1]()
      %4 : None = prim::Constant()
      %5 : Tensor = aten::mean(%0, %2, %3, %4)
      return (%5))IR";
  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenAnyDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=0]()
      %5 : Tensor = aten::any(%0, %1, %3)
      return (%5))IR";
  auto in = at::randint(0, 2, {4, 4, 4}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenAnyDimAllFalseConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=2]()
      %3 : bool = prim::Constant[value=0]()
      %5 : Tensor = aten::any(%0, %1, %3)
      return (%5))IR";
  auto in = at::zeros({3, 7, 4}, at::kCUDA).to(torch::kBool);
  test_body(graph, in);
}

TEST(Converters, ATenAnyDimKeepDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=1]()
      %5 : Tensor = aten::any(%0, %1, %3)
      return (%5))IR";
  auto in = at::randint(0, 2, {4, 4, 4}, at::kCUDA).to(torch::kHalf);
  test_body(graph, in);
}

TEST(Converters, ATenAnyDimNegIndexConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %3 : bool = prim::Constant[value=1]()
      %5 : Tensor = aten::any(%0, %1, %3)
      return (%5))IR";
  std::vector<int> data(64, 0);
  for (int i = 0; i < 64; ++i) {
    if (i % 7 == 0)
      data[i] = 1; // some positives
    if (i % 13 == 0)
      data[i] = -1; // some negatives
  }
  auto in = at::tensor(data, at::TensorOptions().dtype(at::kInt).device(at::kCUDA)).reshape({2, 32}); // shape [2, 32]
  test_body(graph, in);
}

TEST(Converters, ATenAllDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %3 : bool = prim::Constant[value=0]()
      %5 : Tensor = aten::all(%0, %1, %3)
      return (%5))IR";
  auto in = at::randint(0, 2, {64, 2}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenAllDimKeepDimConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=0]()
      %3 : bool = prim::Constant[value=1]()
      %5 : Tensor = aten::all(%0, %1, %3)
      return (%5))IR";
  auto in = at::randint(-2, 2, {2, 32}, at::kCUDA).to(torch::kBool);
  test_body(graph, in);
}

TEST(Converters, ATenAllDimAllTrueConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %3 : bool = prim::Constant[value=0]()
      %5 : Tensor = aten::all(%0, %1, %3)
      return (%5))IR";
  auto in = at::ones({2, 32}, at::kCUDA);
  test_body(graph, in);
}

TEST(Converters, ATenAllDimDynamicConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=-1]()
      %3 : bool = prim::Constant[value=0]()
      %5 : Tensor = aten::all(%0, %1, %3)
      return (%5))IR";
  auto in = at::randint(0, 2, {64, 2}, at::kCUDA).to(torch::kHalf);
  test_body(graph, in, true);
}
