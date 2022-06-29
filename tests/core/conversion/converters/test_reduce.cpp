#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

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

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
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

TEST(Converters, UnpackVarLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedNegAxisLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %37 : bool = prim::Constant[value=1]()
        %53 : int[] = prim::Constant[value=[-1]]()
        %69 : Tensor = aten::var(%x.1, %53, %37, %37)
        return (%69))IR";

  auto in = at::randint(-5, 5, {2, 20, 768}, at::kCUDA).to(at::kFloat);

  auto jit_in = at::clone(in);
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {jit_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}
