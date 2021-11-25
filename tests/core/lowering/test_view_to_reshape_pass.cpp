#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ViewToReshapeCorrectly) {
  std::string source_graph = R"IR(
    graph(%x : Tensor, %1, %1.1):
        %0 : int = prim::Constant[value=0]()
        %2 : Tensor = aten::permute(%x, %1)
        %3 : Tensor = aten::contiguous(%2, %0)
        %4 : Tensor = aten::view(%3, %1.1)
        return (%4))IR";
  std::string target_graph = R"IR(
    graph(%x : Tensor, %1, %1.1):
        %0 : int = prim::Constant[value=0]()
        %2 : Tensor = aten::permute(%x, %1)
        %4 : Tensor = aten::reshape(%2, %1.1)
        return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::RemoveContiguous(sg);
  torch_tensorrt::core::lowering::passes::ViewToReshape(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}
