#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

void pointwise_test_helper(
    std::string graph_ir,
    bool singleInput,
    bool dynamicInput = false,
    std::vector<int64_t> shape1 = {5},
    std::vector<int64_t> shape2 = {5}) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_ir, g.get());

  // singleInput case is enabled when elementwise operation is performed
  // with an input and a constant embedded in graph
  std::vector<at::Tensor> torch_inputs;
  torch_inputs.push_back(at::randint(1, 5, shape1, {at::kCUDA}));
  if (!singleInput) {
    torch_inputs.push_back(at::randint(1, 5, shape2, {at::kCUDA}));
  }
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, torch_inputs);

  std::vector<at::Tensor> trt_inputs;
  for (auto in : torch_inputs) {
    trt_inputs.push_back(at::clone(in));
  }

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  std::vector<at::Tensor> trt_results;
  if (dynamicInput) {
    trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, trt_inputs);
  } else {
    trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, trt_inputs);
  }

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAddConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::add(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenAddWithAlphaConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : float = prim::Constant[value=3.2]()
        %3 : Tensor = aten::add(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenAddImplicitWithAlphaConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : float = prim::Constant[value=7.6]()
        %3 : Tensor = aten::add_(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
}

TEST(Converters, ATenAddWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %scalar : float = prim::Constant[value=2.4]()
        %3 : Tensor = aten::add(%0, %scalar, %2)
        return (%3))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenSubConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=2.3]()
        %3 : Tensor = aten::sub(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenMulConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::mul(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenMulWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::mul(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenDivConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::div(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenDivWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::div(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenPowTensorConvertsCorrectly) {
  const auto graph = R"IR(
       graph(%x.1 : Tensor, %x2.1 : Tensor):
          %3 : Tensor = aten::pow(%x.1, %x2.1)
          return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenPowScalarConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %3 : Tensor = aten::pow(%x.1, %2)
          return (%3))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenNeTensorConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor,
      %y.1 : Tensor):
        %3 : Tensor = aten::ne(%x.1, %y.1)
        return (%3))IR";
  pointwise_test_helper(graph, false, false, {3, 4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4}, {3, 4});
}

TEST(Converters, ATenNeScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=2]()
            %3 : Tensor = aten::ne(%x.1, %2)
            return (%3))IR";
  pointwise_test_helper(graph, true, false, {3, 4, 2});
}

TEST(Converters, ATenFloorDivideConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::floor_divide(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenFloorDivideWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::floor_divide(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenMaxConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::max(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenMinConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::min(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3});
}

TEST(Converters, ATenRsubWithTensorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=2]()
        %3 : Tensor = aten::rsub(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, false, {4}, {3, 4});
  pointwise_test_helper(graph, false, true, {4, 3, 3, 3}, {4, 3, 3, 3});
}

TEST(Converters, ATenRsubWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=2]()
        %scalar : float = prim::Constant[value=2.4]()
        %3 : Tensor = aten::rsub(%0, %scalar, %2)
        return (%3))IR";
  pointwise_test_helper(graph, true, false, {4, 3, 3, 3});
}

TEST(Converters, ATenClampMinConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%x.1 : Tensor):
          %2 : float = prim::Constant[value=1.5]()
          %3 : None = prim::Constant()
          %4 : Tensor = aten::clamp(%x.1, %2, %3)
          return (%4))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenClampMaxConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%x.1 : Tensor):
          %2 : float = prim::Constant[value=3.5]()
          %3 : None = prim::Constant()
          %4 : Tensor = aten::clamp(%x.1, %3, %2)
          return (%4))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenClampMinMaxConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%x.1 : Tensor):
          %2 : float = prim::Constant[value=3.5]()
          %3 : float = prim::Constant[value=1.5]()
          %4 : Tensor = aten::clamp(%x.1, %3, %2)
          return (%4))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenClampMinimumConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%x.1 : Tensor):
          %2 : float = prim::Constant[value=2.5]()
          %4 : Tensor = aten::clamp_min(%x.1, %2)
          return (%4))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenClampMaximumConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%x.1 : Tensor):
          %2 : float = prim::Constant[value=2.5]()
          %4 : Tensor = aten::clamp_max(%x.1, %2)
          return (%4))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenGreaterThanConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor, %1 : Tensor):
      %2 : Tensor = aten::gt(%0, %1)
      return (%2))IR";
  pointwise_test_helper(graph, false, false, {5, 5}, {5, 5});
}

TEST(Converters, ATenGreaterThanScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %scalar : float = prim::Constant[value=3]()
      %2 : Tensor = aten::gt(%0, %scalar)
      return (%2))IR";
  pointwise_test_helper(graph, true, false, {5, 5});
}

TEST(Converters, ATenLessThanConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor, %1 : Tensor):
      %2 : Tensor = aten::lt(%0, %1)
      return (%2))IR";
  pointwise_test_helper(graph, false, false, {5, 5}, {5, 5});
}

TEST(Converters, ATenLessThanScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %scalar : float = prim::Constant[value=3]()
      %2 : Tensor = aten::lt(%0, %scalar)
      return (%2))IR";
  pointwise_test_helper(graph, true, false, {5, 5});
}

TEST(Converters, ATenEqualConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor, %1 : Tensor):
      %2 : Tensor = aten::eq(%0, %1)
      return (%2))IR";
  pointwise_test_helper(graph, false, false, {5, 5}, {5, 5});
}

TEST(Converters, ATenEqualScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %scalar : float = prim::Constant[value=3]()
      %2 : Tensor = aten::eq(%0, %scalar)
      return (%2))IR";
  pointwise_test_helper(graph, true, false, {5, 5});
}

TEST(Converters, ATenGEConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor, %1 : Tensor):
      %2 : Tensor = aten::ge(%0, %1)
      return (%2))IR";
  pointwise_test_helper(graph, false, false, {5, 5}, {5, 5});
}

TEST(Converters, ATenGEScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %scalar : float = prim::Constant[value=3]()
      %2 : Tensor = aten::ge(%0, %scalar)
      return (%2))IR";
  pointwise_test_helper(graph, true, false, {5, 5});
}

TEST(Converters, ATenLEConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor, %1 : Tensor):
      %2 : Tensor = aten::le(%0, %1)
      return (%2))IR";
  pointwise_test_helper(graph, false, false, {5, 5}, {5, 5});
}

TEST(Converters, ATenLEScalarConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%0 : Tensor):
      %scalar : float = prim::Constant[value=3]()
      %2 : Tensor = aten::le(%0, %scalar)
      return (%2))IR";
  pointwise_test_helper(graph, true, false, {5, 5});
}
