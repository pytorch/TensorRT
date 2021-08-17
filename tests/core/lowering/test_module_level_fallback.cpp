#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "core/lowering/lowering.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"

TEST(Lowering, LowerResNet18ModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/resnet18_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::core::ir::Input> input_ranges{trtorch::core::ir::Input({1, 3, 224, 224})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.lower_info.forced_fallback_modules.push_back("torchvision.models.resnet.BasicBlock");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::core::CompileGraph(mod, cfg);

  auto g = trt_mod.get_method("forward").graph();
  auto nodes = g->block()->nodes();
  std::size_t count = 0;
  for (const auto n : nodes) {
    auto has_compile_attribute = n->hasAttribute(c10::Symbol::attr("to_compile"));
    if (has_compile_attribute && n->i(c10::Symbol::attr("to_compile")) == (int64_t) false) {
      count++;
    }
  }
  ASSERT_TRUE(count == 62);

  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(Lowering, LowerAndPartitionSimpleModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/module_fallback_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 1, 16, 16}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::core::ir::Input> input_ranges{trtorch::core::ir::Input({1, 1, 16, 16})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.lower_info.forced_fallback_modules.push_back("ModuleFallbackSub");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::core::CompileGraph(mod, cfg);

  auto g = trt_mod.get_method("forward").graph();
  auto nodes = g->block()->nodes();
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
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}

TEST(Lowering, LowerAndPartitionMobileNetModuleFallbackCorrectly) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_traced.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

  const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  std::vector<trtorch::core::ir::Input> input_ranges{trtorch::core::ir::Input({1, 3, 224, 224})};
  trtorch::core::CompileSpec cfg(input_ranges);
  cfg.partition_info.enabled = true;
  cfg.partition_info.min_block_size = 5;
  cfg.lower_info.forced_fallback_modules.push_back("torchvision.models.mobilenetv2.ConvBNActivation");

  auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
  auto trt_mod = trtorch::core::CompileGraph(mod, cfg);

  auto g = trt_mod.get_method("forward").graph();
  auto nodes = g->block()->nodes();
  std::size_t trt_count = 0;
  std::size_t fallback_count = 0;
  for (const auto n : nodes) {
    auto has_name_attribute = n->hasAttribute(c10::Symbol::attr("name"));
    auto has_compile_attribute = n->hasAttribute(c10::Symbol::attr("to_compile"));
    if (has_name_attribute && n->s(c10::Symbol::attr("name")).find("trt_engine") != std::string::npos) {
      trt_count++;
    } else if (has_compile_attribute && n->i(c10::Symbol::attr("to_compile")) == (int64_t) false) {
      fallback_count++;
    }
  }
  ASSERT_TRUE(trt_count == 1);
  ASSERT_TRUE(fallback_count == 105);

  auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
  ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results, trt_results, 2e-6));
}
