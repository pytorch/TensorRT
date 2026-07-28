#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Evaluators, PrimConstantEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=1]()
        return (%0))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimListUnpackEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %lc : int[] = prim::ListConstruct(%1, %2)
        %lu.1 : int, %lu.2 : int = prim::ListUnpack(%lc)
        return (%lu.1, %lu.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
  ASSERT_TRUE(jit_results[1] == trt_results[1]);
}

TEST(Evaluators, NumToTensorEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %lu.1 : Tensor = prim::NumToTensor(%1)
        return (%lu.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleConstruct1EvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %tc : (int) = prim::TupleConstruct(%1)
        return (%tc))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleConstruct2EvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %tc : (int, int) = prim::TupleConstruct(%1, %2)
        return (%tc))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleConstruct3EvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %3 : int = prim::Constant[value=4]()
        %tc : (int, int, int) = prim::TupleConstruct(%1, %2, %3)
        return (%tc))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleConstruct4EvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %3 : int = prim::Constant[value=3]()
        %4 : int = prim::Constant[value=4]()
        %tc : (int, int, int, int) = prim::TupleConstruct(%1, %2, %3, %4)
        return (%tc))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleUnpackEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %tc : (int, int) = prim::TupleConstruct(%1, %2)
        %tu.1 : int, %tu.2 : int = prim::TupleUnpack(%tc)
        return (%tu.1, %tu.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}

TEST(Evaluators, PrimTupleIndexEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph():
        %0 : int = prim::Constant[value=1]()
        %1 : int = prim::Constant[value=3]()
        %2 : int = prim::Constant[value=4]()
        %tc : (int, int) = prim::TupleConstruct(%1, %2)
        %ti : int = prim::TupleIndex(%tc, %0)
        return (%ti))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {});
  auto trt_results = torch_tensorrt::tests::util::EvaluateGraph(g->block(), {});

  ASSERT_TRUE(jit_results[0] == trt_results[0]);
}