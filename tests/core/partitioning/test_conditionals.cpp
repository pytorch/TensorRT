#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"

size_t count_trt_engines_in_conditionals(std::shared_ptr<torch::jit::Graph> g) {
  size_t count = 0;
  for (auto n : g->nodes()) {
    if (n->kind() == torch::jit::prim::If) {
      std::vector<torch::jit::Block*> blocks{n->blocks()[0], n->blocks()[1]};
      for (auto cur_block : blocks) {
        for (auto n : cur_block->nodes()) {
          if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
            ++count;
          }
        }
      }
    }
  }
  return count;
}

TEST(Partitioning, FallbackOnConditionalsCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/conditional_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  std::vector<trtorch::core::ir::InputRange> input_ranges{trtorch::core::ir::InputRange({3, 3, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = trtorch::core::CompileGraph(mod, cfg);
  auto g = new_mod.get_method("forward").graph();

  auto conditional_engines_count = count_trt_engines_in_conditionals(g);

  ASSERT_TRUE(conditional_engines_count == 2);
}
