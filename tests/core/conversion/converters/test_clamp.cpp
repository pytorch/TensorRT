#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

using torch_tensorrt::tests::util::pointwise_test_helper;

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