#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/torch.h"

TEST(LoweringPasses, UnpackAndCastMaskedFillLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: Tensor, %x.2: Tensor, %x.3: float):
        %2 : Tensor = aten::masked_fill_(%x.1, %x.2, %x.3)
        return (%2))IR";

  auto in = at::rand({2, 3, 5, 7}, {at::kCUDA});
  auto in2 = at::rand({2, 3, 5, 7}, {at::kCUDA}).to(torch::kBool);
  auto in3 = 7.3;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in, in2, in3});
  torch_tensorrt::core::lowering::passes::UnpackAndCastMaskedFill(g, "cuda:0");
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in, in2, in3});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackAndCastNumToTensorLowersIntCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: int):
        %2 : Tensor = prim::NumToTensor(%x.1)
        return (%2))IR";

  auto in = 1;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackAndCastNumToTensor(g, "cuda:0");
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackAndCastNumToTensorLowersFloatCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: float):
        %2 : Tensor = prim::NumToTensor(%x.1)
        return (%2))IR";

  auto in = 78.1;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackAndCastNumToTensor(g, "cuda:0");
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackAndCastFullIntLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: int):
        %5 : NoneType = prim::Constant()
        %2 : int = prim::Constant[value=3]()
        %10 : int[] = prim::ListConstruct(%2, %2)
        %out : Tensor = aten::full(%10, %x.1, %5, %5, %5, %5)
        return (%out))IR";

  auto in = 4;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackAndCastFull(g, "cuda:0");
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      jit_pre_results[0].toTensor(), jit_post_results[0].toTensor().cpu(), 2e-6));
}

TEST(LoweringPasses, UnpackAndCastFullFloatLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: float):
        %5 : NoneType = prim::Constant()
        %2 : int = prim::Constant[value=5]()
        %3 : int = prim::Constant[value=4]()
        %10 : int[] = prim::ListConstruct(%2, %3)
        %out : Tensor = aten::full(%10, %x.1, %5, %5, %5, %5)
        return (%out))IR";

  auto in = 54.1;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackAndCastFull(g, "cuda:0");
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      jit_pre_results[0].toTensor(), jit_post_results[0].toTensor().cpu(), 2e-6));
}

TEST(LoweringPasses, ReplaceScalarImplicitLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: Tensor):
        %5 : int = prim::Constant[value=0]()
        %false : bool = prim::Constant[value=0]()
        %none : NoneType = prim::Constant()
        %cuda : Device = prim::Constant[value="cuda:0"]()
        %3 : int = aten::size(%x.1, %5)
        %y.2 : Tensor = prim::NumToTensor(%3)
        %y.1 : Tensor = aten::to(%y.2, %cuda, %none, %false, %false)
        %19 : Tensor[] = prim::ListConstruct(%x.1, %y.1)
        %21 : Tensor, %22 : Tensor = prim::ListUnpack(%19)
        %2 : Scalar = aten::ScalarImplicit(%22)
        %out : Tensor = prim::NumToTensor(%2)
        return (%out))IR";

  auto in = at::rand({2, 3, 5, 7}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::ReplaceScalarImplicit(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, ReplaceScalarImplicitIntNumToTensorLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: int):
        %1 : Tensor = prim::NumToTensor(%x.1)
        %2 : Scalar = aten::ScalarImplicit(%1)
        %3 : Tensor = prim::NumToTensor(%2)
        return (%3))IR";

  auto in = 25;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackAndCastNumToTensor(g, "cuda:0");
  torch_tensorrt::core::lowering::passes::ReplaceScalarImplicit(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, ReplaceScalarImplicitFloatLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1: float):
        %1 : Tensor = prim::NumToTensor(%x.1)
        %2 : Scalar = aten::ScalarImplicit(%1)
        %3 : Tensor = prim::NumToTensor(%2)
        return (%3))IR";

  auto in = 2.5;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::ReplaceScalarImplicit(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}
