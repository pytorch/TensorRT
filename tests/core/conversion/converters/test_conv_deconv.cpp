#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

// aten::_convolution(Tensor input, Tensor weight,
//                    Tensor? bias, int[] stride, int[] padding,
//                    int[] dilation, bool transposed,
//                    int[] output_padding, int groups, bool benchmark,
//                    bool deterministic, bool cudnn_enabled) -> (Tensor)

void conv_test_helper(std::string graph_ir) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_ir, g.get());

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});
  auto w = at::randint(1, 10, {8, 3, 5, 5}, {at::kCUDA});
  auto b = at::randint(1, 10, {8}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolutionConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 3, 5, 5, strides=[45, 15, 5, 1]),
            %2 : Float(8)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 10, 10}, {at::kCUDA});
  auto w = at::randint(1, 10, {8, 3, 5, 5}, {at::kCUDA});
  auto b = at::randint(1, 10, {8}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolution1dConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1])):
        %2 : None = prim::Constant()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3)
        %9 : int[] = prim::ListConstruct(%4)
        %10 : int[] = prim::ListConstruct(%5)
        %11 : int[] = prim::ListConstruct(%6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {1, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConv1dWithWeightTensorsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 5, 3, strides=[15, 3, 1])):
        %2 : int = prim::Constant[value=-128]()
        %3 : float = prim::Constant[value=3.5]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=127]()
        %quant_input : Tensor = aten::fake_quantize_per_tensor_affine(%0, %3, %4, %2, %5)
        %6 : int = prim::Constant[value=6]()
        %7 : int = prim::Constant[value=4]()
        %8 : Device = prim::Constant[value="cuda:0"]()
        %9 : None = prim::Constant()
        %10 : int[] = prim::ListConstruct(%7)
        %11 : Tensor = aten::full(%10, %3, %6, %9, %8, %9)
        %12 : int[] = prim::ListConstruct(%7)
        %13 : int = prim::Constant[value=0]()
        %14 : Tensor = aten::full(%12, %13, %6, %9, %8, %9)
        %quant_wts : Tensor = aten::fake_quantize_per_channel_affine(%1, %11, %14, %13, %2, %5)
        %15 : None = prim::Constant()
        %16 : int = prim::Constant[value=1]()
        %17 : int = prim::Constant[value=0]()
        %18 : int = prim::Constant[value=1]()
        %19 : int = prim::Constant[value=0]()
        %20 : bool = prim::Constant[value=0]()
        %21 : int[] = prim::ListConstruct(%16)
        %22 : int[] = prim::ListConstruct(%17)
        %23 : int[] = prim::ListConstruct(%18)
        %24 : int[] = prim::ListConstruct(%19)
        %25 : Tensor = aten::_convolution(%quant_input, %quant_wts, %15, %21, %22, %23, %20, %24, %16, %20, %20, %20, %20)
        return (%25))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 5, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 5, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in}, nvinfer1::DataType::kINT8);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenConvolutionNoBiasConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 1, 3, 3, strides=[9, 9, 3, 1])):
        %2 : None = prim::Constant()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {1, 1, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 1, 2, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolutionWithStrideConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, 3, strides=[27, 9, 3, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=3]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 9, 9}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolutionWithPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 4, 4, strides=[48, 16, 4, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=2]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 2, 2}, {at::kCUDA});
  auto b = at::randint(1, 10, {4}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolution3dConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(32, 3, 3, 3, 3, strides=[81, 27, 9, 3, 1]),
            %2 : Float(32)):
        %sv : int = prim::Constant[value=1]()
        %s : int[] = prim::ListConstruct(%sv, %sv, %sv)
        %pv : int = prim::Constant[value=0]()
        %p : int[] = prim::ListConstruct(%pv, %pv, %pv)
        %transposed : bool = prim::Constant[value=0]()
        %opv : int = prim::Constant[value=0]()
        %op : int[] = prim::ListConstruct(%opv, %opv, %opv)
        %g : int = prim::Constant[value=1]()
        %fb : bool = prim::Constant[value=0]()
        %out : Tensor = aten::_convolution(%0, %1, %2, %s, %p, %s, %transposed, %op, %g, %fb, %fb, %fb, %fb)
        return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 10, {32, 3, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolution3dNoBiasConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(32, 3, 3, 3, 3, strides=[81, 27, 9, 3, 1])):
        %bias : None = prim::Constant()
        %sv : int = prim::Constant[value=1]()
        %s : int[] = prim::ListConstruct(%sv, %sv, %sv)
        %pv : int = prim::Constant[value=0]()
        %p : int[] = prim::ListConstruct(%pv, %pv, %pv)
        %transposed : bool = prim::Constant[value=0]()
        %opv : int = prim::Constant[value=0]()
        %op : int[] = prim::ListConstruct(%opv, %opv, %opv)
        %g : int = prim::Constant[value=1]()
        %fb : bool = prim::Constant[value=0]()
        %out : Tensor = aten::_convolution(%0, %1, %bias, %s, %p, %s, %transposed, %op, %g, %fb, %fb, %fb, %fb)
        return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {1, 3, 5, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 2, {32, 3, 3, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolution3dWithPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(32, 3, 3, 3, 3, strides=[81, 27, 9, 3, 1]),
            %2 : Float(32)):
        %sv : int = prim::Constant[value=1]()
        %s : int[] = prim::ListConstruct(%sv, %sv, %sv)
        %pv : int = prim::Constant[value=1]()
        %p : int[] = prim::ListConstruct(%pv, %pv, %pv)
        %transposed : bool = prim::Constant[value=0]()
        %opv : int = prim::Constant[value=0]()
        %op : int[] = prim::ListConstruct(%opv, %opv, %opv)
        %g : int = prim::Constant[value=1]()
        %fb : bool = prim::Constant[value=0]()
        %out : Tensor = aten::_convolution(%0, %1, %2, %s, %p, %s, %transposed, %op, %g, %fb, %fb, %fb, %fb)
        return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 10, {32, 3, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolution3dWithStrideDilationConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(32, 3, 3, 3, 3, strides=[81, 27, 9, 3, 1]),
            %2 : Float(32)):
        %sv : int = prim::Constant[value=2]()
        %s : int[] = prim::ListConstruct(%sv, %sv, %sv)
        %pv : int = prim::Constant[value=1]()
        %p : int[] = prim::ListConstruct(%pv, %pv, %pv)
        %transposed : bool = prim::Constant[value=0]()
        %opv : int = prim::Constant[value=0]()
        %op : int[] = prim::ListConstruct(%opv, %opv, %opv)
        %g : int = prim::Constant[value=1]()
        %fb : bool = prim::Constant[value=0]()
        %out : Tensor = aten::_convolution(%0, %1, %2, %s, %p, %s, %transposed, %op, %g, %fb, %fb, %fb, %fb)
        return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 10, {32, 3, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTransposeConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 3, 3, 3, strides=[27, 9, 3, 1]),
            %2 : Float(8)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 3, {1, 8, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 3, {8, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 3, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTranspose2dWithWeightsAsTensorsConvertsCorrectly) {
  const auto graph = R"IR(
  graph(%0 : Tensor,
        %1 : Float(48, 56, 3, 3, strides=[504, 9, 3, 1])):
    %2 : int = prim::Constant[value=-128]()
    %3 : float = prim::Constant[value=3.5]()
    %4 : int = prim::Constant[value=0]()
    %5 : int = prim::Constant[value=127]()
    %quant_input : Tensor = aten::fake_quantize_per_tensor_affine(%0, %3, %4, %2, %5)
    %6 : int = prim::Constant[value=6]()
    %7 : int = prim::Constant[value=56]()
    %8 : Device = prim::Constant[value="cuda:0"]()
    %9 : None = prim::Constant()
    %10 : int[] = prim::ListConstruct(%7)
    %11 : Tensor = aten::full(%10, %3, %6, %9, %8, %9)
    %12 : int[] = prim::ListConstruct(%7)
    %13 : int = prim::Constant[value=1]()
    %14 : Tensor = aten::full(%12, %13, %6, %9, %8, %9)
    %quant_wts : Tensor = aten::fake_quantize_per_channel_affine(%1, %11, %14, %13, %2, %5)
    %15 : None = prim::Constant()
    %16 : bool = prim::Constant[value=1]()
    %17 : int = prim::Constant[value=1]() # Adjusted padding
    %17.1: int = prim::Constant[value=0]() # Adjusted out_padding
    %18 : int = prim::Constant[value=1]() # Adjusted dilation
    %19 : int = prim::Constant[value=2]() # Adjusted stride
    %20 : int = prim::Constant[value=1]()
    %21 : int[] = prim::ListConstruct(%17)
    %22 : int[] = prim::ListConstruct(%17, %17)
    %23 : int[] = prim::ListConstruct(%18, %18)
    %23.1: int[] = prim::ListConstruct(%17.1, %17.1)
    %24 : int[] = prim::ListConstruct(%19, %19)
    %25 : Tensor = aten::_convolution(%quant_input, %quant_wts, %15, %24, %22, %23, %16, %23.1, %17, %16, %16, %16, %16)
    return (%25))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 48, 2, 200}, {at::kCUDA});
  auto w = at::randint(1, 2, {48, 56, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in, jit_w});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in, trt_w}, nvinfer1::DataType::kINT8);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenConvTransposeNoBiasConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 1, 3, 3, strides=[9, 9, 3, 1])):
        %2 : None = prim::Constant()
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %3, %7, %7, %7, %7)
        return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {1, 4, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {4, 1, 2, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTransposeWithStrideConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, 3, strides=[27, 9, 3, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=3]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 4, 9, 9}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTransposeWithPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 4, 4, strides=[48, 16, 4, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=2]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 4, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 2, 2}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConv1dTransposeWithPaddingOutPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 3, strides=[9, 3, 1])):
        %2 : None = prim::Constant()
        %3 : int = prim::Constant[value=2]()
        %4 : int = prim::Constant[value=1]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3)
        %9 : int[] = prim::ListConstruct(%4)
        %10 : int[] = prim::ListConstruct(%5)
        %11 : int[] = prim::ListConstruct(%6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {1, 3, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {3, 4, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConv1dTransposeWithWeightTensorsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 5, 3, strides=[15, 3, 1])):
        %2 : int = prim::Constant[value=-128]()
        %3 : float = prim::Constant[value=3.5]()
        %4 : int = prim::Constant[value=0]()
        %5 : int = prim::Constant[value=127]()
        %quant_input : Tensor = aten::fake_quantize_per_tensor_affine(%0, %3, %4, %2, %5)
        %6 : int = prim::Constant[value=6]()
        %7 : int = prim::Constant[value=4]()
        %8 : Device = prim::Constant[value="cuda:0"]()
        %9 : None = prim::Constant()
        %10 : int[] = prim::ListConstruct(%7)
        %11 : Tensor = aten::full(%10, %3, %6, %9, %8, %9)
        %12 : int[] = prim::ListConstruct(%7)
        %13 : int = prim::Constant[value=1]()
        %14 : Tensor = aten::full(%12, %13, %6, %9, %8, %9)
        %quant_wts : Tensor = aten::fake_quantize_per_channel_affine(%1, %11, %14, %13, %2, %5)
        %15 : None = prim::Constant()
        %16 : int = prim::Constant[value=1]()
        %17 : int = prim::Constant[value=0]()
        %18 : int = prim::Constant[value=1]()
        %19 : int = prim::Constant[value=0]()
        %20 : bool = prim::Constant[value=0]()
        %21 : int[] = prim::ListConstruct(%16)
        %22 : int[] = prim::ListConstruct(%17)
        %23 : int[] = prim::ListConstruct(%18)
        %24 : int[] = prim::ListConstruct(%19)
        %25 : bool = prim::Constant[value=1]()
        %26 : Tensor = aten::_convolution(%quant_input, %quant_wts, %15, %21, %22, %23, %25, %24, %18, %20, %20, %20, %20)
        return (%26))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {4, 5, 3}, {at::kCUDA});
  auto w = at::randint(1, 2, {5, 4, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in}, nvinfer1::DataType::kINT8);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0], 2e-6));
}

TEST(Converters, ATenConvTransposeWithPaddingOutPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 4, 4, strides=[48, 16, 4, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=2]()
        %4 : int = prim::Constant[value=2]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=1]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 4, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 2, 2}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0];

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTransposeOutPaddingBiggerThanPaddingConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(4, 3, 4, 4, strides=[48, 16, 4, 1]),
            %2 : Float(4)):
        %3 : int = prim::Constant[value=4]()
        %4 : int = prim::Constant[value=2]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=3]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=1]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 4, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 10, {4, 3, 2, 2}, {at::kCUDA});
  auto b = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvolutionWithGroupConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 1, 2, 2, strides=[48, 16, 4, 1]),
            %2 : Float(8)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=2]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=0]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=4]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 4, 4, 4}, {at::kCUDA});
  auto w = at::randint(1, 10, {8, 1, 2, 2}, {at::kCUDA});
  auto b = at::randint(1, 10, {8}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenConvTransposeWithGroupConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Float(8, 4, 3, 3, strides=[56, 16, 3, 1]),
            %2 : Float(16)):
        %3 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=1]()
        %5 : int = prim::Constant[value=1]()
        %6 : int = prim::Constant[value=0]()
        %7 : bool = prim::Constant[value=1]()
        %8 : int[] = prim::ListConstruct(%3, %3)
        %9 : int[] = prim::ListConstruct(%4, %4)
        %10 : int[] = prim::ListConstruct(%5, %5)
        %11 : int[] = prim::ListConstruct(%6, %6)
        %12 : int = prim::Constant[value=4]()
        %13 : Tensor = aten::_convolution(%0, %1, %2, %8, %9, %10, %7, %11, %12, %7, %7, %7)
        return (%13))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 8, 5, 5}, {at::kCUDA});
  auto w = at::randint(1, 10, {8, 4, 3, 3}, {at::kCUDA});
  auto b = at::randint(1, 10, {16}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_w = at::clone(w);
  auto jit_b = at::clone(b);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {jit_w, jit_b});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_w = at::clone(w);
  auto trt_b = at::clone(b);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {trt_w, trt_b});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}
