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

TEST(Evaluators, OnesEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant() # :0:0
        %3 : int[] = aten::size(%x.1) # <string>:7:9
        %z.1 : Tensor = aten::ones(%3, %2, %2, %2, %2) # experiments/test_zeros.py:8:12
        return (%z.1))IR";

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
}

TEST(Evaluators, OnesDataTypeEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=5]() # :0:0 (Float16)
        %3 : None = prim::Constant() # :0:0
        %4 : int[] = aten::size(%x.1) # <string>:7:9
        %z.1 : Tensor = aten::ones(%4, %2, %3, %3, %3) # experiments/test_zeros.py:8:12
        return (%z.1))IR";

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
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

TEST(Evaluators, ATenSizeNegativeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=-1]()
        %2 : int = prim::Constant[value=-2]()
        %3 : int = aten::size(%0, %1)
        %4 : int = aten::size(%0, %2)
        %5 : int[] = prim::ListConstruct(%3, %4)
        %6 : Tensor = aten::view(%0, %5)
        return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
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

TEST(Evaluators, ATenAppendWithITensorEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : Tensor[] = prim::ListConstruct(%0)
        %4 : Tensor[] = aten::append(%3, %1)
        %5 : Tensor = aten::cat(%4, %2)
        return (%5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in0 = at::randint(1, 10, {3, 3}, {at::kCUDA});
  auto in1 = at::randint(1, 10, {3, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in0, in1});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in0, in1});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Evaluators, ATenAppendWithTensorEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[3,3]]()
        %2 : None = prim::Constant() # :0:0
        %20 : Device = prim::Constant[value="cuda"]()
        %3 : Tensor = aten::zeros(%1, %2, %2, %20, %2)
        %4 : Tensor = aten::zeros(%1, %2, %2, %20, %2)
        %5 : int = prim::Constant[value=0]()
        %15 : int = prim::Constant[value=1]()
        %6 : Tensor[] = prim::ListConstruct(%3)
        %7 : Tensor[] = aten::append(%6, %4)
        %8 : Tensor = aten::cat(%7, %5)
        %9 : Tensor = aten::add(%8, %0, %15)
        return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in0 = at::randint(1, 10, {6, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in0});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in0});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Evaluators, ATenAppendWithITensorAndTensorEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = aten::size(%0)
        %2 : None = prim::Constant() # :0:0
        %20 : Device = prim::Constant[value="cuda"]()
        %3 : Tensor = aten::zeros(%1, %2, %2, %20, %2)
        %4 : int = prim::Constant[value=0]()
        %5 : Tensor[] = prim::ListConstruct(%0)
        %6 : Tensor[] = aten::append(%5, %3)
        %7 : Tensor = aten::cat(%6, %4)
        return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in0 = at::randint(1, 10, {3, 3}, {at::kCUDA});

  auto params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto jit_results = trtorch::tests::util::RunGraph(g, params, {in0});

  params = trtorch::core::conversion::get_named_params(g->inputs(), {});
  auto trt_results = trtorch::tests::util::RunGraphEngine(g, params, {in0});

  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Evaluators, ATenCloneEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %4 : None = prim::Constant()
        %5 : Tensor = aten::clone(%0, %4)
        return (%5))IR";

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
}


TEST(Evaluators, ATenCopyEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=3]()
        %3 : int = prim::Constant[value=10]()
        %4 : int = prim::Constant[value=10]()
        %5 : int[] = prim::ListConstruct(%1, %2, %3, %4)
        %6 : None = prim::Constant()
        %7 : Device = prim::Constant[value="cuda"]()
        %8 : Tensor = aten::ones(%5, %6, %6, %7, %6)
        %9 : bool = prim::Constant[value=0]()
        %10 : Tensor = aten::copy_(%8, %0, %9)
        return (%10))IR";

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_results = trtorch::tests::util::EvaluateGraphJIT(g, {in});
  auto trt_results = trtorch::tests::util::EvaluateGraph(g->block(), {in});

  ASSERT_TRUE(at::equal(jit_results[0].toTensor().to(at::kCUDA), trt_results[0].toTensor()));
}