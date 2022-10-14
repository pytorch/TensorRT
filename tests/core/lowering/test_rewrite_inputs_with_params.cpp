#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, RewriteInputsWithParamsCorrectly) {
  std::string source_graph = R"IR(
    graph(%x: Tensor, %y: Tensor, %1 : Int(1)):
        %out: Tensor = aten::sub(%x, %y, %1)
        return (%out))IR";
  std::string target_graph = R"IR(
    graph(%x: Tensor,  %y : Tensor):
        %2 : int = prim::Constant[value=0]()
        %out: Tensor = aten::sub(%x, %y, %2)
        return (%out))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  torch::jit::IValue param0 = torch::jit::IValue(0);
  std::vector<torch::jit::IValue> params{param0};
  torch_tensorrt::core::lowering::passes::RewriteInputsWithParams(sg, params);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}