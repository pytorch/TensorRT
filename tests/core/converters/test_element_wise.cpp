#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

void pointwise_test_helper(
    std::string graph_ir,
    bool singleInput,
    std::vector<int64_t> shape1 = {5},
    std::vector<int64_t> shape2 = {5}) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_ir, &*g);

  // singleInput case is enabled when elementwise operation is performed
  // with an input and a constant embedded in graph
  std::vector<at::Tensor> torch_inputs;
  torch_inputs.push_back(at::randint(1, 5, shape1, {at::kCUDA}));
  if (!singleInput) {
    torch_inputs.push_back(at::randint(1, 5, shape2, {at::kCUDA}));
  }
  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, torch_inputs);

  std::vector<at::Tensor> trt_inputs;
  for (auto in : torch_inputs) {
    trt_inputs.push_back(at::clone(in));
  }

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, trt_inputs);

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenAddConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::add(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, {4}, {3, 4});
}

TEST(Converters, ATenSubConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Tensor = aten::sub(%0, %1, %2)
        return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, {4}, {3, 4});
}

TEST(Converters, ATenMulConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::mul(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, {4}, {3, 4});
}

TEST(Converters, ATenDivConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor = aten::div(%0, %1)
        return (%2))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, {4}, {3, 4});
}

TEST(Converters, ATenPowTensorConvertsCorrectly) {
  const auto graph = R"IR(
       graph(%x.1 : Tensor, %x2.1 : Tensor):
          %3 : Tensor = aten::pow(%x.1, %x2.1)
          return (%3))IR";
  pointwise_test_helper(graph, false);
  pointwise_test_helper(graph, false, {3, 4}, {4});
  pointwise_test_helper(graph, false, {4}, {3, 4});
}

TEST(Converters, ATenPowScalarConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %3 : Tensor = aten::pow(%x.1, %2)
          return (%3))IR";
  pointwise_test_helper(graph, true);
}
