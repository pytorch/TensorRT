#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

using torch_tensorrt::tests::util::pointwise_test_helper;

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