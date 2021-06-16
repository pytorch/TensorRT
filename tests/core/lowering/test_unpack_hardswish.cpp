#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, UnpackHardSwish) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardswish(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%input):
            %1 : Scalar = prim::Constant[value=3.]()
            %2 : Scalar = prim::Constant[value=1.]()
            %3 = aten::add(%input, %1, %2)
            %4 : Scalar = prim::Constant[value=0.]()
            %5 : Scalar = prim::Constant[value=6.]()
            %6 = aten::hardtanh(%3, %4, %5)
            %7 = aten::div(%6, %5)
            %8 = aten::mul(%input, %7)
            return (%8))IR";

  trtorch::core::util::logging::get_logger().set_reportable_log_level(trtorch::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  trtorch::core::lowering::passes::UnpackHardSwish(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, UnpackHardInplaceSwish) {
  std::string source_graph = R"IR(
        graph(%input):
            %result = aten::hardswish_(%input)
            return (%result))IR";

  std::string target_graph = R"IR(
        graph(%input):
            %1 : Scalar = prim::Constant[value=3.]()
            %2 : Scalar = prim::Constant[value=1.]()
            %3 = aten::add(%input, %1, %2)
            %4 : Scalar = prim::Constant[value=0.]()
            %5 : Scalar = prim::Constant[value=6.]()
            %6 = aten::hardtanh(%3, %4, %5)
            %7 = aten::div(%6, %5)
            %8 = aten::mul(%input, %7)
            return (%8))IR";

  trtorch::core::util::logging::get_logger().set_reportable_log_level(trtorch::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  trtorch::core::lowering::passes::UnpackHardSwish(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}