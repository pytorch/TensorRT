#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenResizeDynamicShapeCorrectly) {
  const auto graph = R"IR(
    graph(%x : Tensor):
          %3 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=-1]()
          %28 : int = aten::size(%x, %3)
          %30 : int[] = prim::ListConstruct(%28, %2)
          %6 : Tensor = aten::reshape(%x, %30)
          return (%6))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 3, 2}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true, /*allow_shape_tensors=*/true);

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenResizeDynamicInputCorrectly) {
  const auto graph = R"IR(
    graph(%x : Tensor):
          %2 : int[] = prim::Constant[value=[-1, 4, 64]]()
          %3 : Tensor = aten::reshape(%x, %2)
          return (%3))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 16, 16}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true, /*allow_shape_tensors=*/true);

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenResizeGetItemDynShapeCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %3 : int = prim::Constant[value=-1]()
            %2 : int = prim::Constant[value=0]()
            %size.1 : int[] = aten::size(%x.1)
            %37 : int = aten::__getitem__(%size.1, %2)
            %39 : int[] = prim::ListConstruct(%37, %3)
            %7 : Tensor = aten::reshape(%x.1, %39)
            return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 16, 16}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true, /*allow_shape_tensors=*/true);

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenResizeGetItemDynShapeMulCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %2 : int = prim::Constant[value=0]()
            %3 : int = prim::Constant[value=-1]()
            %4 : int = prim::Constant[value=2]()
            %size.1 : int[] = aten::size(%x.1)
            %37 : int = aten::__getitem__(%size.1, %2)
            %38 : int = aten::mul(%37, %4)
            %39 : int[] = prim::ListConstruct(%38, %3)
            %7 : Tensor = aten::reshape(%x.1, %39)
            return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 16, 16}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results =
      torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {in}, true, /*allow_shape_tensors=*/true);

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}