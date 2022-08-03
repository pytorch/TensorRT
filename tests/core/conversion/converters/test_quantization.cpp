#include <string>
#include "NvInfer.h"
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenFakeQuantizePerTensorConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %7 : int = prim::Constant[value=-128]()
      %3 : float = prim::Constant[value=6.]()
      %4 : int = prim::Constant[value=0]()
      %8 : int = prim::Constant[value=127]()
      %quant_input.1 : Tensor = aten::fake_quantize_per_tensor_affine(%x.1, %3, %4, %7, %8)
      return (%quant_input.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 5, 5}, {at::kCUDA}).to(at::kFloat);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in}, nvinfer1::DataType::kINT8);

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenFakeQuantizePerChannelConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
        %22 : int = prim::Constant[value=-128]()
        %14 : int = prim::Constant[value=4]()
        %9 : None = prim::Constant()
        %35 : Device = prim::Constant[value="cuda:0"]()
        %6 : int = prim::Constant[value=6]()
        %3 : int = prim::Constant[value=5]()
        %5 : float = prim::Constant[value=3.5]()
        %13 : int = prim::Constant[value=1]()
        %23 : int = prim::Constant[value=127]()
        %4 : int[] = prim::ListConstruct(%3)
        %11 : Tensor = aten::full(%4, %5, %6, %9, %35, %9)
        %12 : int[] = prim::ListConstruct(%3)
        %19 : Tensor = aten::full(%12, %13, %14, %9, %35, %9)
        %quant_input.1 : Tensor = aten::fake_quantize_per_channel_affine(%x.1, %11, %19, %13, %22, %23)
        return (%quant_input.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 5, 3, 3}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in}, nvinfer1::DataType::kINT8);

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
