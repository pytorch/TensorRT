#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include "torch/csrc/jit/passes/canonicalize.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

TEST(LoweringPasses, Conv1dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1]),
            %2 : Float(4)):
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=1]()
        %stride : int[] = prim::ListConstruct(%6)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)
        %12 : Tensor = aten::conv1d(%0, %1, %2, %stride, %padding, %dilation, %6)
        return (%12))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1]),
            %2 : Float(4)):
        %3 : bool = prim::Constant[value=0]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=1]()
        %stride : int[] = prim::ListConstruct(%6)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)
        %output_padding : int[] = prim::Constant[value=[0]]()
        %12 : Tensor = aten::_convolution(%0, %1, %2, %stride, %padding, %dilation, %3, %output_padding, %6, %3, %3, %3, %3)
        return (%12))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::Conv1DToConvolution(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 2, {1, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, ConvTransposed1dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 3, 3, strides=[9, 3, 1]),
            %2 : Float(3)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %stride : int[] = prim::ListConstruct(%3)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)
        %output_padding : int[] = prim::ListConstruct(%6)
        %12 : Tensor = aten::conv_transpose1d(%0, %1, %2, %stride, %padding, %output_padding, %3, %dilation)
        return (%12))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 3, 3, strides=[9, 3, 1]),
            %2 : Float(3)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : bool = prim::Constant[value=1]()
        %stride : int[] = prim::ListConstruct(%3)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)
        %output_padding : int[] = prim::ListConstruct(%6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %stride, %padding, %dilation, %8, %output_padding, %5, %7, %7, %7, %7)
        return (%12))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ConvTransposed1DToConvolution(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 2, {1, 8, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {8, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, Conv2dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, strides=[36, 9, 3, 1]),
            %2 : Float(3)):
        %16 : int[] = prim::Constant[value=[0, 0]]()
        %15 : int[] = prim::Constant[value=[1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %11 : Tensor = aten::conv2d(%0, %1, %2, %15, %16, %15, %5)
        return (%11))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, strides=[36, 9, 3, 1]),
            %2 : Float(3)):
        %3 : int[] = prim::Constant[value=[0, 0]]()
        %4 : int[] = prim::Constant[value=[1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::Constant[value=[0, 0]]()
        %9 : Tensor = aten::_convolution(%0, %1, %2, %4, %3, %4, %7, %8, %5, %7, %7, %7, %7)
        return (%9))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::Conv2DToConvolution(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));

  auto in = at::randint(1, 2, {3, 4, 4, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {3, 4, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, ConvTransposed2dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, strides=[36, 9, 3, 1]),
            %2 : Float(4)):
        %93 : int[] = prim::Constant[value=[0, 0]]()
        %92 : int[] = prim::Constant[value=[1, 1]]()
        %12 : int = prim::Constant[value=1]()
        %88 : Tensor = aten::conv_transpose2d(%0, %1, %2, %92, %93, %93, %12, %92)
        return (%88))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, strides=[36, 9, 3, 1]),
            %2 : Float(4)):
        %3 : int[] = prim::Constant[value=[0, 0]]()
        %4 : int[] = prim::Constant[value=[1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=1]()
        %8 : bool = prim::Constant[value=1]()
        %9 : Tensor = aten::_convolution(%0, %1, %2, %4, %3, %4, %7, %3, %5, %8, %8, %8, %8)
        return (%9))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ConvTransposed2DToConvolution(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));

  auto in = at::randint(1, 2, {3, 3, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {3, 4, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, Conv3dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, 3, strides=[108, 27, 9, 3, 1]),
            %2 : Float(3)):
        %16 : int[] = prim::Constant[value=[0, 0, 0]]()
        %15 : int[] = prim::Constant[value=[1, 1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %11 : Tensor = aten::conv3d(%0, %1, %2, %15, %16, %15, %5)
        return (%11))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, 3, strides=[108, 27, 9, 3, 1]),
            %2 : Float(3)):
        %3 : int[] = prim::Constant[value=[0, 0, 0]]()
        %4 : int[] = prim::Constant[value=[1, 1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::Constant[value=[0, 0, 0]]()
        %9 : Tensor = aten::_convolution(%0, %1, %2, %4, %3, %4, %7, %8, %5, %7, %7, %7, %7)
        return (%9))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::Conv3DToConvolution(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));

  auto in = at::randint(1, 2, {4, 4, 4, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 2, {3, 4, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, ConvTransposed3dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, 3, strides=[108, 27, 9, 3, 1]),
            %2 : Float(4)):
        %93 : int[] = prim::Constant[value=[0, 0, 0]]()
        %92 : int[] = prim::Constant[value=[1, 1, 1]]()
        %13 : int = prim::Constant[value=1]()
        %88 : Tensor = aten::conv_transpose3d(%0, %1, %2, %92, %93, %93, %13, %92)
        return (%88))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(3, 4, 3, 3, 3, strides=[108, 27, 9, 3, 1]),
            %2 : Float(4)):
        %3 : int[] = prim::Constant[value=[0, 0, 0]]()
        %4 : int[] = prim::Constant[value=[1, 1, 1]]()
        %5 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=1]()
        %8 : bool = prim::Constant[value=1]()
        %9 : Tensor = aten::_convolution(%0, %1, %2, %4, %3, %4, %7, %3, %5, %8, %8, %8, %8)
        return (%9))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ConvTransposed3DToConvolution(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));

  auto in = at::randint(1, 2, {3, 3, 3, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {3, 4, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, Conv1dWithConditionalLowersCorrectly) {
  std::string source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1]),
            %2 : Float(3)):
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=1]()
        %stride : int[] = prim::ListConstruct(%6)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)

        # Add intentionally-invalid weight tensor to ensure prim::If blocks are respected
        %true : bool = prim::Constant[value=1]()
        %invalid_weight : Tensor = aten::transpose(%0, %4, %5)
        %12 : Tensor = prim::If(%true)
            block0():
                %res: Tensor = aten::conv1d(%0, %1, %2, %stride, %padding, %dilation, %6)
                -> (%res)
            block1():
                %res: Tensor = aten::conv1d(%invalid_weight, %1, %2, %stride, %padding, %dilation, %6)
                -> (%res)
        return (%12))IR";

  std::string target_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1]),
            %2 : Float(3)):
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %true : bool = prim::Constant[value=1]()
        %3 : bool = prim::Constant[value=0]()
        %output_padding : int[] = prim::Constant[value=[0]]()
        %6 : int = prim::Constant[value=1]()
        %stride : int[] = prim::ListConstruct(%6)
        %padding : int[] = prim::ListConstruct(%4)
        %dilation : int[] = prim::ListConstruct(%5)

        # Add intentionally-invalid weight tensor to ensure prim::If blocks are respected
        %invalid_weight : Tensor = aten::transpose(%0, %4, %5)
        %12 : Tensor = prim::If(%true)
            block0():
                %res: Tensor = aten::_convolution(%0, %1, %2, %stride, %padding, %dilation, %3, %output_padding, %6, %3, %3, %3, %3)
                -> (%res)
            block1():
                %res: Tensor = aten::_convolution(%invalid_weight, %1, %2, %stride, %padding, %dilation, %3, %output_padding, %6, %3, %3, %3, %3)
                -> (%res)
        return (%12))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::Conv1DToConvolution(sg);
  torch::jit::ConstantPooling(sg);
  sg = torch::jit::Canonicalize(sg, false);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);
  torch::jit::ConstantPooling(tg);
  tg = torch::jit::Canonicalize(tg, false);

  // Validate identical graphs after pooling constants and canonicalizing
  ASSERT_TRUE((tg->toString() == sg->toString()));

  auto in = at::randint(1, 2, {1, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {trt_w, trt_b});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {trt_w, trt_b});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}
