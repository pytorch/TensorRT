#include <string>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenEinsumConvertsMatMulCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor, %x.2 : Tensor):
        %0 : str = prim::Constant[value="ij,jk->ik"]()
        %3 : Tensor[] = prim::ListConstruct(%x.1, %x.2)
        %4 : Tensor = aten::einsum(%0, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Test matrix multiplication via einsum
  auto in_0 = at::rand({12, 17}, {at::kCUDA});
  auto in_1 = at::rand({17, 35}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenEinsumConvertsElementwiseProdCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor, %x.2 : Tensor):
        %0 : str = prim::Constant[value="abcd,abcd->abcd"]()
        %3 : Tensor[] = prim::ListConstruct(%x.1, %x.2)
        %4 : Tensor = aten::einsum(%0, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Test elementwise tensor product via einsum
  auto in_0 = at::rand({7, 5, 2, 8}, {at::kCUDA});
  auto in_1 = at::rand({7, 5, 2, 8}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenEinsumConvertsTransposeCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %0 : str = prim::Constant[value="jk->kj"]()
        %3 : Tensor[] = prim::ListConstruct(%x.1)
        %4 : Tensor = aten::einsum(%0, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Test single-matrix transpose via einsum
  auto in_0 = at::rand({25, 28}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenEinsumConvertsVectorsCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor, %x.2 : Tensor):
        %0 : str = prim::Constant[value="a,b->ab"]()
        %3 : Tensor[] = prim::ListConstruct(%x.1, %x.2)
        %4 : Tensor = aten::einsum(%0, %3)
        return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  // Test vector outer product via einsum
  auto in_0 = at::rand({25}, {at::kCUDA});
  auto in_1 = at::rand({4}, {at::kCUDA});

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in_0, in_1});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in_0, in_1});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}