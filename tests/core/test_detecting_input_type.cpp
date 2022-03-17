#include <string>
#include "core/ir/ir.h"
#include "core/lowering/lowering.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

TEST(CoreTest, DetectingInputTypeWorksCorrectFP32) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  auto graph_and_parameters = torch_tensorrt::core::lowering::Lower(mod, "forward", {});
  auto g = graph_and_parameters.first;

  auto input_types = torch_tensorrt::core::ir::get_block_first_calc_dtypes_opt(g->block());

  for (auto in : input_types) {
    c10::optional<at::ScalarType>& detected_type_opt = in.second;
    ASSERT_TRUE(detected_type_opt);
    ASSERT_TRUE(detected_type_opt.value() == at::kFloat);
  }
}

TEST(CoreTest, DetectingInputTypeWorksCorrectFP16) {
  torch::jit::script::Module mod;
  try {
    mod = torch::jit::load("tests/modules/mobilenet_v2_scripted.jit.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    ASSERT_TRUE(false);
  }

  mod.to(at::kHalf);

  auto graph_and_parameters = torch_tensorrt::core::lowering::Lower(mod, "forward", {});
  auto g = graph_and_parameters.first;

  auto input_types = torch_tensorrt::core::ir::get_block_first_calc_dtypes_opt(g->block());

  for (auto in : input_types) {
    c10::optional<at::ScalarType>& detected_type_opt = in.second;
    ASSERT_TRUE(detected_type_opt);
    ASSERT_TRUE(detected_type_opt.value() == at::kHalf);
  }
}
