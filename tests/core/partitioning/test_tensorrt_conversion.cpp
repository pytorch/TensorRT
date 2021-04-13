#include <string>
#include "core/compiler.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

int count_trt_engines(std::shared_ptr<torch::jit::Graph> g) {
  int count = 0;
  for (const auto n : g->nodes()) {
    if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
      ++count;
    }
  }
  return count;
}

TEST(Partitioning, ConvertResNet50SegmentedBlockCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet50_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({1, 3, 224, 224})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("aten::add");
  torch::jit::script::Module new_mod = trtorch::core::CompileGraph(mod, cfg);
  auto g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(g);
  ASSERT_TRUE(count == 17);
}

TEST(Partitioning, ConvertMobileNetSegmentedBlockCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({1, 3, 224, 224})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("aten::add");
  torch::jit::script::Module new_mod = trtorch::core::CompileGraph(mod, cfg);
  auto g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(g);
  ASSERT_TRUE(count == 11);
}
