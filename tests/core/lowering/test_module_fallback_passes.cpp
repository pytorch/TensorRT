#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "core/lowering/lowering.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/script.h"

TEST(Lowering, NotateModuleForFallbackWorksCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/module_fallback_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  std::unordered_set<std::string> mods_to_mark;
  mods_to_mark.insert("custom_models.ModuleFallbackSub");

  torch_tensorrt::core::lowering::passes::NotateModuleForFallback(mod, "", "forward", mods_to_mark);

  auto g = mod.get_method("forward").graph();
  auto nodes = g->block()->nodes();

  bool seen_enter = false;
  int64_t enter_count = 0;
  int64_t exit_count = 0;
  int64_t intermediate_nodes = 0;
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto n = *it;
    if (n->kind() == torch::jit::prim::Enter) {
      enter_count++;
      auto internal_n = *(++it);
      ASSERT_TRUE(internal_n->kind() != torch::jit::prim::Exit);
      intermediate_nodes++;
      auto end = *(++it);
      ASSERT_TRUE(end->kind() == torch::jit::prim::Exit);
      exit_count++;
      seen_enter = true;
    }
  }
  ASSERT_TRUE(seen_enter);
  ASSERT_TRUE(enter_count == 1);
  ASSERT_TRUE(intermediate_nodes == 1);
  ASSERT_TRUE(exit_count == 1);
}

TEST(Lowering, MarkNodesForFallbackWorksCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/module_fallback_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  std::unordered_set<std::string> mods_to_mark;
  mods_to_mark.insert("custom_models.ModuleFallbackSub");

  torch_tensorrt::core::lowering::passes::NotateModuleForFallback(mod, "", "forward", mods_to_mark);
  auto mod_ = torch::jit::freeze_module(mod);
  auto g = mod_.get_method("forward").graph();
  torch_tensorrt::core::lowering::passes::MarkNodesForFallback(g, true);
  auto nodes = g->block()->nodes();

  int64_t num_marked_nodes = 0;

  for (auto n : nodes) {
    auto has_compile_attribute = n->hasAttribute(c10::Symbol::attr("to_compile"));
    if (has_compile_attribute && n->i(c10::Symbol::attr("to_compile")) == (int64_t) false) {
      num_marked_nodes++;
    }
  }

  ASSERT_TRUE(num_marked_nodes == 2);
}

TEST(Lowering, LowerAndPartitionSimpleModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/module_fallback_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 1, 16, 16}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto g = mod.get_method("forward").graph();

  std::vector<torch_tensorrt::core::ir::Input> input_ranges{torch_tensorrt::core::ir::Input({1, 1, 16, 16})};
  torch_tensorrt::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.lower_info.forced_fallback_modules.push_back("ModuleFallbackSub");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);

  auto trt_g = trt_mod.get_method("forward").graph();
  auto nodes = trt_g->block()->nodes();
  std::size_t curr_node = 0;
  for (const auto n : nodes) {
    if (curr_node == 5) {
      ASSERT_TRUE(n->kind() == torch::jit::aten::conv2d);
      ASSERT_TRUE(n->i(c10::Symbol::attr("to_compile")) == (int64_t) false);
    } else if (curr_node == 6) {
      ASSERT_TRUE(n->kind() == torch::jit::aten::relu);
      ASSERT_TRUE(n->i(c10::Symbol::attr("to_compile")) == (int64_t) false);
    } else if (curr_node == 7) {
      ASSERT_TRUE(n->kind() == torch::jit::prim::GetAttr);
      ASSERT_TRUE(n->s(c10::Symbol::attr("name")).find("trt_engine") != std::string::npos);
    }
    curr_node++;
  }

  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
