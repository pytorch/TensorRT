#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RemoveDropoutLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::dropout(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveDropoutInplaceLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::dropout_(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveFeatureDropoutLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::feature_dropout(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveFeatureDropoutInplaceLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::feature_dropout_(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveFeatureAlphaDropoutLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::feature_alpha_dropout(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveFeatureAlphaDropoutInplaceLowersCorrectly) {
  std::string source_graph = R"IR(
    graph(%x.1):
      %3 : float = prim::Constant[value=0.5]()
      %4 : bool = prim::Constant[value=0]()
      %y.1 : Tensor = aten::feature_alpha_dropout_(%x.1, %3, %4)
      %11 : Tensor = aten::relu(%y.1)
      return (%11))IR";
  std::string target_graph = R"IR(
    graph(%x.1):
      %11 : Tensor = aten::relu(%x.1)
      return (%11))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveDropout(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
