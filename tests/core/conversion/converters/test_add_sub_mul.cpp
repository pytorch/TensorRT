#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

using torch_tensorrt::tests::util::pointwise_test_helper;

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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
}

TEST(Converters, ATenAddInplaceWithAlphaConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : float = prim::Constant[value=7.6]()
        %3 : Tensor = aten::add_(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3});
  pointwise_test_helper(graph, false, false, {3, 4, 3}, {4, 3}, false, at::kFloat, at::kInt);
}

TEST(Converters, ATenAddImplicitWithIntAlphaConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=42]()
        %3 : Tensor = aten::add_(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false, false, {2, 2}, {2, 2}, false, at::kInt, at::kInt);
  pointwise_test_helper(graph, false, false, {3, 4, 3}, {4, 3}, false, at::kInt, at::kInt);
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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
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
  pointwise_test_helper(graph, false, false, {4, 3, 3, 3}, {4, 3, 3, 3}, false, at::kInt, at::kFloat);
  pointwise_test_helper(graph, false, false, {4, 3, 3, 3}, {4, 3, 3, 3}, false, at::kInt, at::kInt);
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

TEST(Converters, ATenRsubWithIntScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %2 : int = prim::Constant[value=2]()
        %scalar : int = prim::Constant[value=8]()
        %3 : Tensor = aten::rsub(%0, %scalar, %2)
        return (%3))IR";
  pointwise_test_helper(graph, true, false, {4, 3, 3, 3}, {}, false, at::kInt);
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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
}

TEST(Converters, ATenSquareConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor = aten::square(%0)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenMulWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::mul(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenMulWithIntScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : int = prim::Constant[value=2]()
        %1 : Tensor = aten::mul(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true, false, {5}, {5}, false, at::kInt);
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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
}

TEST(Converters, ATenPowScalarConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %3 : Tensor = aten::pow(%x.1, %2)
          return (%3))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ElementWiseTypePromotionDisambiguatesCastNames) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::add(%0, %1, %2)
        %4 : Tensor = aten::add(%0, %1, %2)
        %5 : Tensor = aten::add(%3, %4, %2)
        return (%5))IR";
  pointwise_test_helper(graph, false, false, {4, 3, 3, 3}, {4, 3, 3, 3}, false, at::kInt, at::kFloat);
}
