#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/torch.h"

using torch_tensorrt::tests::util::pointwise_test_helper;

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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kInt);
}

TEST(Converters, ATenDivWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::div(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
}

TEST(Converters, ATenDivWithScalarIntConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : int = prim::Constant[value=2]()
        %1 : Tensor = aten::div(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
  pointwise_test_helper(graph, true, false, {5}, {1}, false, at::kInt);
}

TEST(Converters, ATenDivRoundingFloorConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %3 : str = prim::Constant[value="floor"]()
        %2 : Tensor = aten::div(%0, %1, %3)
        return (%2))IR";
  pointwise_test_helper(graph, false, false, {5}, {5}, true);
  pointwise_test_helper(graph, false, false, {3, 4}, {4}, true);
  pointwise_test_helper(graph, false, false, {4}, {3, 4}, true);
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3}, true);
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3}, true);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kInt);
}

TEST(Converters, ATenDivRoundingTruncConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %3 : str = prim::Constant[value="trunc"]()
        %2 : Tensor = aten::div(%0, %1, %3)
        return (%2))IR";
  pointwise_test_helper(graph, false, false, {5}, {5}, true);
  pointwise_test_helper(graph, false, false, {3, 4}, {4}, true);
  pointwise_test_helper(graph, false, false, {4}, {3, 4}, true);
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3}, true);
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3}, true);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kInt);
}

TEST(Converters, ATenDivRoundingNoneConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %3 : None = prim::Constant()
        %2 : Tensor = aten::div(%0, %1, %3)
        return (%2))IR";
  pointwise_test_helper(graph, false, false, {5}, {5}, true);
  pointwise_test_helper(graph, false, false, {3, 4}, {4}, true);
  pointwise_test_helper(graph, false, false, {4}, {3, 4}, true);
  pointwise_test_helper(graph, false, true, {3, 4, 3}, {4, 3}, true);
  pointwise_test_helper(graph, false, true, {4, 3}, {3, 4, 3}, true);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kInt);
}

TEST(Converters, ATenDivRoundingTruncWithIntsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %trunc : str = prim::Constant[value="trunc"]()
        %out : Tensor = aten::div(%0, %1, %trunc)
        return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Avoid divide-by-zero issues by making denominator >= 1
  auto in_0 = at::randint(-5, 5, {4, 1, 7, 8}, {at::kCUDA}).to(torch::kInt32);
  auto in_1 = at::randint(1, 10, {4, 1, 7, 8}, {at::kCUDA}).to(torch::kInt32);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0])));
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
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kFloat, at::kInt);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kFloat);
  pointwise_test_helper(graph, false, true, {5}, {5}, false, at::kInt, at::kInt);
}

TEST(Converters, ATenFloorDivideWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::floor_divide(%0, %scalar)
        return (%1))IR";
  pointwise_test_helper(graph, true);
  pointwise_test_helper(graph, true, false, {5}, {5}, false, at::kInt);
}

TEST(Converters, ATenRemainderConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::remainder(%0, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto input1 = at::randint(-5, 5, {4, 5}, {at::kCUDA});
  auto input2 = at::randint(1, 5, {5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {input1, input2});

  torch_tensorrt::core::lowering::passes::ReduceRemainder(g);

  input1 = at::clone(input1);
  input2 = at::clone(input2);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {input1, input2});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 5e-2));
}

TEST(Converters, ATenRemainderWithScalarConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %scalar : float = prim::Constant[value=2.4]()
        %1 : Tensor = aten::remainder(%0, %scalar)
        return (%1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(-5, 5, {5}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  torch_tensorrt::core::lowering::passes::ReduceRemainder(g);

  in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}