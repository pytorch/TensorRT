#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

TEST(LoweringPasses, RemoveUnnecessaryCastIntCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: int):
      %2: Tensor = aten::NumToTensor(%1)
      %3: int = aten::Int(%2)
      %4: int = aten::add(%3, %3, %3)
      return (%4))IR";
  std::string target_graph = R"IR(
    graph(%1: int):
      %4: int = aten::add(%1, %1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveUnnecessaryCastFloatCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: float):
      %2: Tensor = aten::NumToTensor(%1)
      %3: float = aten::Float(%2)
      %4: float = aten::add(%3, %3, %3)
      return (%3))IR";
  std::string target_graph = R"IR(
    graph(%1: float):
      %4: float = aten::add(%1, %1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveUnnecessaryCastBoolCorrectly) {
  std::string source_graph = R"IR(
    graph(%1: bool):
      %2: Tensor = aten::NumToTensor(%1)
      %3: bool = aten::Bool(%2)
      %4: bool = aten::__and__(%3, %3)
      return (%3))IR";
  std::string target_graph = R"IR(
    graph(%1: bool):
      %4: bool = aten::__and__(%1, %1)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());
  torch_tensorrt::core::lowering::passes::RemoveUnnecessaryCasts(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsIntCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[8]]()
      %2: int = prim::Constant[value=1]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::add(%1, %3, %2)
      %5: int = aten::Int(%4)
      %6: int = aten::add(%5, %5)
      return (%6))IR";
  std::string target_graph = R"IR(
    graph(%0: int):
      %1: int = prim::Constant[value=8]()
      %4: int = aten::add(%1, %0)
      %6: int = aten::add(%4, %4)
      return (%6))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(8), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloatCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: float):
      %1: Tensor = prim::Constant[value=[8.]]()
      %2: float = prim::Constant[value=1.]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::add(%1, %3, %2)
      %5: float = aten::Float(%4)
      %6: float = aten::add(%5, %5)
      return (%6))IR";
  std::string target_graph = R"IR(
    graph(%0: float):
      %1: float = prim::Constant[value=8.]()
      %4: float = aten::add(%1, %0)
      %6: float = aten::add(%4, %4)
      return (%6))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(8.0), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloorDivIntCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[7]]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::floor_divide(%1, %3)
      %5: int = aten::Int(%4)
      return (%5))IR";
  std::string target_graph = R"IR(
    graph(%0: int):
      %1: int = prim::Constant[value=7]()
      %4: int = aten::floordiv(%1, %0)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(7), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloorDivFloatCorrectly) {
  std::string source_graph = R"IR(
    graph(%0: float):
      %1: Tensor = prim::Constant[value=[8.]]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::floor_divide(%1, %3)
      %5: float = aten::Float(%4)
      return (%5))IR";
  std::string target_graph = R"IR(
    graph(%0: float):
      %1: float = prim::Constant[value=8.]()
      %4: float = aten::floordiv(%1, %0)
      return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::WithInsertPoint guard(first_op);
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(8.0), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloorDivIntValuesAgree) {
  std::string source_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %11: int = prim::Constant[value=7]()
      %3: Tensor = prim::NumToTensor(%0)
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::floor_divide(%1, %3)
      %50: int = aten::Int(%4)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %1: int = prim::Constant[value=7]()
      %40: int = aten::floordiv(%1, %0)
      %4: Tensor = prim::NumToTensor(%40)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor()));
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsDivTruncIntValuesAgree) {
  // Ensure the source and target graphs have equivalent outputs
  // (Source and Target are computing equivalent values)
  std::string source_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %11: int = prim::Constant[value=-3]()
      %234 : str = prim::Constant[value="trunc"]()
      %3: Tensor = prim::NumToTensor(%0)
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::div(%1, %3, %234)
      %50: int = aten::Int(%4)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %1: int = prim::Constant[value=-3]()
      %40: float = aten::div(%1, %0)
      %41: int = aten::Int(%40)
      %4: Tensor = prim::NumToTensor(%41)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor()));

  // Ensure the lowering pass transforms the first graph into the second
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[8]]()
      %3: Tensor = prim::NumToTensor(%0)
      %234: str = prim::Constant[value="trunc"]()
      %4: Tensor = aten::div(%3, %1, %234)
      %5: int = aten::Int(%4)
      return (%5))IR";

  std::string target_graph = R"IR(
    graph(%0 : int):
      %1 : str = prim::Constant[value="trunc"]()
      %2 : int = prim::Constant[value=8]()
      %3 : float = aten::div(%0, %2)
      %4 : int = aten::Int(%3)
      return (%4))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(8), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsDivFloorIntValuesAgree) {
  // Ensure the source and target graphs have equivalent outputs
  // (Source and Target are computing equivalent values)
  std::string source_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %11: int = prim::Constant[value=-3]()
      %234 : str = prim::Constant[value="floor"]()
      %3: Tensor = prim::NumToTensor(%0)
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::div(%1, %3, %234)
      %50: int = aten::Int(%4)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %1: int = prim::Constant[value=-3]()
      %40: int = aten::floordiv(%1, %0)
      %41: int = aten::Int(%40)
      %4: Tensor = prim::NumToTensor(%41)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {});

  ASSERT_TRUE(torch_tensorrt::tests::util::exactlyEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor()));

  // Ensure the lowering pass transforms the first graph into the second
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[8]]()
      %3: Tensor = prim::NumToTensor(%0)
      %234: str = prim::Constant[value="floor"]()
      %4: Tensor = aten::div(%3, %1, %234)
      %5: int = aten::Int(%4)
      return (%5))IR";

  std::string target_graph = R"IR(
    graph(%0 : int):
      %1 : str = prim::Constant[value="floor"]()
      %2 : int = prim::Constant[value=8]()
      %3 : int = aten::floordiv(%0, %2)
      return (%3))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  auto first_op = *(sg->block()->nodes().begin());
  torch::jit::Value* r = sg->insertConstant(c10::scalar_to_tensor(8), c10::nullopt, first_op->scope());
  r->copyMetadata(first_op->output());
  r->setType(c10::TensorType::get());
  first_op->output()->replaceAllUsesWith(r);
  first_op->destroy();

  torch_tensorrt::core::lowering::passes::RemoveSingleUse0DTensors(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));
}

TEST(LoweringPasses, RemoveSingleUse0DTensorsFloorDivFloatValuesAgree) {
  std::string source_graph_no_inputs = R"IR(
    graph():
      %0: float = prim::Constant[value=2.]()
      %11: float = prim::Constant[value=7.]()
      %3: Tensor = prim::NumToTensor(%0)
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::floor_divide(%1, %3)
      %50: float = aten::Float(%4)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph():
      %0: float = prim::Constant[value=2.]()
      %1: float = prim::Constant[value=7.]()
      %40: float = aten::floordiv(%1, %0)
      %4: Tensor = prim::NumToTensor(%40)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, RemoveAtenIntTensorValuesAgree) {
  std::string source_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %11: int = prim::Constant[value=7]()
      %3: Tensor = prim::NumToTensor(%0)
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::floor_divide(%1, %3)
      %7: Tensor = aten::mul(%3, %4)
      %8: Tensor = aten::mul(%7, %1)
      %50: int = aten::Int(%8)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph():
      %0: int = prim::Constant[value=2]()
      %1: int = prim::Constant[value=7]()
      %4: int = aten::floordiv(%1, %0)
      %7: int = aten::mul(%0, %4)
      %40: int = aten::mul(%7, %1)
      %4: Tensor = prim::NumToTensor(%40)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));

  // Ensure the lowering pass transforms the first graph into the second
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph_no_inputs, sg.get());

  torch_tensorrt::core::lowering::passes::ReplaceAtenInt(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph_no_inputs, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveAtenIntSizeTensorValuesAgree) {
  std::string source_graph_no_inputs = R"IR(
    graph(%x.0: Tensor):
      %10: int = prim::Constant[value=0]()
      %100: int = aten::size(%x.0, %10)
      %0: Tensor = prim::NumToTensor(%100)
      %11: int = prim::Constant[value=9]()
      %1: Tensor = prim::NumToTensor(%11)
      %4: Tensor = aten::floor_divide(%1, %0)
      %7: Tensor = aten::mul(%0, %4)
      %8: Tensor = aten::mul(%7, %1)
      %50: int = aten::Int(%8)
      %5: Tensor = prim::NumToTensor(%50)
      return (%5))IR";
  std::string target_graph_no_inputs = R"IR(
    graph(%x.0: Tensor):
      %10: int = prim::Constant[value=0]()
      %0: int = aten::size(%x.0, %10)
      %1: int = prim::Constant[value=9]()
      %4: int = aten::floordiv(%1, %0)
      %7: int = aten::mul(%0, %4)
      %40: int = aten::mul(%7, %1)
      %4: Tensor = prim::NumToTensor(%40)
      return (%4))IR";

  auto g_in = std::make_shared<torch::jit::Graph>();
  auto g_out = std::make_shared<torch::jit::Graph>();

  auto in_0 = at::rand({2, 3, 5, 5}, {at::kCUDA});

  torch::jit::parseIR(source_graph_no_inputs, g_in.get());
  torch::jit::parseIR(target_graph_no_inputs, g_out.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_in, {in_0});
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g_out, {in_0});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));

  // Ensure the lowering pass transforms the first graph into the second
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph_no_inputs, sg.get());

  torch_tensorrt::core::lowering::passes::ReplaceAtenInt(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph_no_inputs, tg.get());

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, RemoveAtenIntConstTensorValuesAgree) {
  // Ensure the lowering pass transforms the first graph into the second
  std::string source_graph = R"IR(
    graph(%0: int):
      %1: Tensor = prim::Constant[value=[8]]()
      %3: Tensor = prim::NumToTensor(%0)
      %4: Tensor = aten::floor_divide(%3, %1)
      %5: int = aten::Int(%4)
      return (%5))IR";

  std::string target_graph = R"IR(
    graph(%0 : int):
      %1 : Tensor = prim::Constant[value=[8]]()
      %2 : int = prim::Constant[value=8]()
      %3 : int = aten::floordiv(%0, %2)
      return (%3))IR";

  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);

  // Manually enter 0d tensor const for source
  auto first_op_sg = *(sg->block()->nodes().begin());
  torch::jit::Value* r_sg = sg->insertConstant(c10::scalar_to_tensor(8), c10::nullopt, first_op_sg->scope());
  r_sg->copyMetadata(first_op_sg->output());
  r_sg->setType(c10::TensorType::get());
  first_op_sg->output()->replaceAllUsesWith(r_sg);
  first_op_sg->destroy();

  torch_tensorrt::core::lowering::passes::ReplaceAtenInt(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  // Manually enter 0d tensor const for target
  auto first_op_tg = *(tg->block()->nodes().begin());
  torch::jit::Value* r_tg = tg->insertConstant(c10::scalar_to_tensor(8), c10::nullopt, first_op_tg->scope());
  r_tg->copyMetadata(first_op_tg->output());
  r_tg->setType(c10::TensorType::get());
  first_op_tg->output()->replaceAllUsesWith(r_tg);
  first_op_tg->destroy();

  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));
}

TEST(LoweringPasses, RemoveCollectionCastTuple) {
  // Ensure the lowering pass transforms the first graph into the second
  std::string source_graph = R"IR(
    graph(%x.1 : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %a.1 : Tensor = aten::mul(%x.1, %2)
      %b.1 : Tensor = aten::add(%a.1, %2, %3)
      %c.1 : Tensor = aten::relu(%b.1)
      %d.1 : Tensor = aten::sqrt(%c.1)
      %8 : (Tensor, Tensor, Tensor) = prim::TupleConstruct(%c.1, %d.1, %b.1)
      return (%8))IR";

  std::string target_graph = R"IR(
    graph(%x.1 : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %a.1 : Tensor = aten::mul(%x.1, %2)
      %b.1 : Tensor = aten::add(%a.1, %2, %3)
      %c.1 : Tensor = aten::relu(%b.1)
      %d.1 : Tensor = aten::sqrt(%c.1)
      return (%c.1, %d.1, %b.1))IR";

  // Ensure the lowering pass transforms the first graph into the second
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  torch_tensorrt::core::lowering::passes::RemoveCollectionCast(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));
}

TEST(LoweringPasses, RemoveCollectionCastList) {
  // Ensure the lowering pass transforms the first graph into the second
  std::string source_graph = R"IR(
    graph(%x.1 : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %a.1 : Tensor = aten::mul(%x.1, %2)
      %b.1 : Tensor = aten::add(%a.1, %2, %3)
      %c.1 : Tensor = aten::relu(%b.1)
      %d.1 : Tensor = aten::sqrt(%c.1)
      %8 : (Tensor, Tensor, Tensor) = prim::ListConstruct(%b.1, %c.1, %d.1)
      return (%8))IR";

  std::string target_graph = R"IR(
    graph(%x.1 : Tensor):
      %3 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %a.1 : Tensor = aten::mul(%x.1, %2)
      %b.1 : Tensor = aten::add(%a.1, %2, %3)
      %c.1 : Tensor = aten::relu(%b.1)
      %d.1 : Tensor = aten::sqrt(%c.1)
      return (%b.1, %c.1, %d.1))IR";

  // Ensure the lowering pass transforms the first graph into the second
  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, sg.get());

  torch_tensorrt::core::lowering::passes::RemoveCollectionCast(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, tg.get());

  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));
}
