#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/torch.h"

TEST(Converters, UnpackVarLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackStdUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %one : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%3, %one)
        %7 : Tensor = aten::std(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randint(-5, 5, {4, 4, 4}, at::kCUDA);

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in});
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, UnpackVarUnbiasedNegAxisLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %37 : bool = prim::Constant[value=1]()
        %53 : int[] = prim::Constant[value=[-1]]()
        %69 : Tensor = aten::var(%x.1, %53, %37, %37)
        return (%69))IR";

  auto in = at::randint(-5, 5, {2, 20, 768}, at::kCUDA).to(at::kFloat);

  auto jit_in = at::clone(in);
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  in = at::clone(in);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {jit_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}