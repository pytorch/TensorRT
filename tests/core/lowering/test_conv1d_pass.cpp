#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, Conv1dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1]),
            %2 : Float(3)):
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
            %2 : Float(3)):
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
