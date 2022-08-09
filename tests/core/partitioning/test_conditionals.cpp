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

  std::vector<torch_tensorrt::core::ir::Input> inputs{torch_tensorrt::core::ir::Input({3, 3, 16, 16})};
  auto g = mod.get_method("forward").graph();
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto new_g = new_mod.get_method("forward").graph();

  auto conditional_engines_count = count_trt_engines_in_conditionals(new_g);

  ASSERT_TRUE(conditional_engines_count == 2);
}

TEST(Partitioning, FallbackInplaceOPInConditionalsCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/inplace_op_if_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{4, 4}, {4, 4}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<torch_tensorrt::core::ir::Input> inputs{
      torch_tensorrt::core::ir::Input({4, 4}), torch_tensorrt::core::ir::Input({4, 4})};
  auto g = mod.get_method("forward").graph();
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("prim::ListConstruct");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
