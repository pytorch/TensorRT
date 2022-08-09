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
