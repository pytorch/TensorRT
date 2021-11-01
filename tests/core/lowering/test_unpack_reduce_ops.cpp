#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

TEST(LoweringPasses, UnpackVarLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}
