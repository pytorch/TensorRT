#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Evaluators, DivIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=9]()
        %2 : int = prim::Constant[value=4]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, DivFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : float = prim::Constant[value=9.1]()
        %2 : float = prim::Constant[value=4.2]()
        %3 : float = aten::div(%1, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, ZerosEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant() # :0:0
        %3 : int[] = aten::size(%x.1) # <string>:7:9
        %z.1 : Tensor = aten::zeros(%3, %2, %2, %2, %2) # experiments/test_zeros.py:8:12
        return (%z.1))IR";

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
}

TEST(Evaluators, ZerosDataTypeEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=5]() # :0:0 (Float16)
        %3 : None = prim::Constant() # :0:0
        %4 : int[] = aten::size(%x.1) # <string>:7:9
        %z.1 : Tensor = aten::zeros(%4, %2, %3, %3, %3) # experiments/test_zeros.py:8:12
        return (%z.1))IR";

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
}

TEST(Evaluators, ATenArangeIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=51]()
        %1 : None = prim::Constant()
        %2 : Tensor = aten::arange(%0, %1, %1, %1, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : float = prim::Constant[value=51.2]()
        %1 : None = prim::Constant()
        %2 : Tensor = aten::arange(%0, %1, %1, %1, %1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeStartEndIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=1]()
        %1 : int = prim::Constant[value=51]()
        %2 : None = prim::Constant()
        %3 : Tensor = aten::arange(%0, %1, %2, %2, %2, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeStartEndFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : float = prim::Constant[value=1.5]()
        %1 : float = prim::Constant[value=51.2]()
        %2 : None = prim::Constant()
        %3 : Tensor = aten::arange(%0, %1, %2, %2, %2, %2)
        return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeStartEndStepIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=1]()
        %1 : int = prim::Constant[value=51]()
        %2 : int = prim::Constant[value=1]()
        %3 : None = prim::Constant()
        %4 : Tensor = aten::arange(%0, %1, %2, %3, %3, %3, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, ATenArangeStartEndStepFloatEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : float = prim::Constant[value=1.2]()
        %1 : float = prim::Constant[value=51.6]()
        %2 : float = prim::Constant[value=1.5]()
        %3 : None = prim::Constant()
        %4 : Tensor = aten::arange(%0, %1, %2, %3, %3, %3, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0].toTensor(), trt_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, FloorIntIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=9]()
        %2 : int = aten::floor(%1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, FloorFloatIntEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : float = prim::Constant[value=9.3]()
        %2 : int = aten::floor(%1)
        return (%2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}