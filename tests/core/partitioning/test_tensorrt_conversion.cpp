#include <string>
#include "gtest/gtest.h"
#include "torch/script.h"
#include "core/compiler.h"
#include "core/util/trt_util.h"

int count_trt_engines(std::shared_ptr<torch::jit::Graph> g) {
  int count = 0;
  for (const auto n : g->nodes()) {
    if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
      ++count;
    }
  }
  return count;
}

TEST(Partitioning, ConvertSegmentedBlockCorrectly) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_base_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = trtorch::core::CompileGraph(mod, cfg);
  auto g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(g);
  ASSERT_TRUE(count == 2);
}


TEST(Partitioning, ConvertSegmentedBlockCorrectlyEdge) {
  torch::jit::script::Module mod;
  try {
  mod = torch::jit::load("tests/core/partitioning/test_edge_model.jit");
  } catch (const c10::Error& e) {
  std::cerr << "error loading the model\n";
  return;
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = trtorch::core::CompileGraph(mod, cfg);
  auto g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(g);
  ASSERT_TRUE(count == 2);
}
