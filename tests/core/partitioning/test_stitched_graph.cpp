#include <string>
#include <unordered_set>
#include "gtest/gtest.h"
#include "torch/script.h"
#include "core/compiler.h"
#include "core/util/trt_util.h"

bool checkAllInputsExistInStitchedGraph(std::shared_ptr<torch::jit::Graph> g) {
  std::unordered_set<torch::jit::Value*> available_values;
  for (auto v : g->inputs()) {
    available_values.insert(v);
  }
  for (const auto n : g->nodes()) {
    for (auto input : n->inputs()) {
      if (!available_values.count(input))
        return false;
    }
    for (auto output : n->outputs()) {
      available_values.insert(output);
    }
  }
  return true;
}

TEST(Partitioning, StitchSegmentedBlockCorrectly) {
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
  ASSERT_TRUE(checkAllInputsExistInStitchedGraph(g));
}


TEST(Partitioning, StitchSegmentedBlockCorrectlyEdge) {
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
  ASSERT_TRUE(checkAllInputsExistInStitchedGraph(g));
}
