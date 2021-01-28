#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Evaluators, IntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %lu.1 : int = aten::Int(%0)
        return (%lu.1))IR";

  const std::vector<std::vector<int64_t>> input_shapes = {{1}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues1, jit_inputs_ivalues2;
  std::vector<torch::jit::IValue> trt_inputs_ivalues1, trt_inputs_ivalues2;
  for (auto in_shape : input_shapes) {
    auto in1 = at::rand(in_shape, {at::kCUDA});
    jit_inputs_ivalues1.push_back(in1.clone());
    trt_inputs_ivalues1.push_back(in1.clone());
    auto in2 = at::randint(-10, 10, in_shape, {at::kCUDA});
    jit_inputs_ivalues2.push_back(in2.clone());
    trt_inputs_ivalues2.push_back(in2.clone());
  }

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results1 = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues1);
  auto trt_results1 = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues1);
  ASSERT_TRUE(jit_results1[0] == trt_results1[0]);

  auto jit_results2 = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues2);
  auto trt_results2 = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues2);
  ASSERT_TRUE(jit_results2[0] == trt_results2[0]);
}

TEST(Evaluators, IsFloatingPointEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %lu.1 : bool = aten::is_floating_point(%0)
        return (%lu.1))IR";

  const std::vector<std::vector<int64_t>> input_shapes = {{10, 10}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::rand(in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, jit_inputs_ivalues);
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), trt_inputs_ivalues);

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}