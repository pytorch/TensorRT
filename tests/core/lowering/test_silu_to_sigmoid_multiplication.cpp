#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RemoveSiluLowersCorrectly) {
  std::string source_graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : Tensor = aten::silu(%x.1)
        return (%2))IR";
  std::string target_graph = R"IR(
      graph(%x.1):
        %2 : Tensor = aten::sigmoid(%x.1)
        %3 : Tensor = aten::mul(%x.1, %2)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::SiluToSigmoidMultipication(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
