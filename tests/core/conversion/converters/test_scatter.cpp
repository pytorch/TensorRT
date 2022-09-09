#include <torch/torch.h>
#include <string>
#include <vector>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

TEST(Converter, ATenIndexPutOneDimensionConvertCorrectly) {
  const auto graph = R"IR(
    graph(%b.1 : Tensor,
          %a.1 : Tensor,
          %idx1.1 : Tensor):
          %7 : int = prim::Constant[value=4]()
          %8 : bool = prim::Constant[value=0]()
          %9 : NoneType = prim::Constant()
          %10 : bool = prim::Constant[value=1]()
          %idx10.1 : Tensor = aten::to(%idx1.1, %7, %8, %8, %9)
          %15 : Tensor?[] = prim::ListConstruct(%idx10.1)
          %16 : Tensor = aten::index_put(%b.1, %15, %a.1, %10)
          return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {13}, {at::kCUDA}).to(at::kFloat);
  auto in1 = torch::tensor({6, 5, 6, 6, 5, 7, 11}, {at::kCUDA}).to(torch::kFloat32);
  auto in2 = torch::tensor({-2, 3, -3, -4, 0, -1, 1}, {at::kCUDA}).to(torch::kInt32);

  auto jit_in = at::clone(in);
  auto jit_in1 = at::clone(in1);
  auto jit_in2 = at::clone(in2);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in, jit_in1, jit_in2});

  auto trt_in = at::clone(in);
  auto trt_in1 = at::clone(in1);
  auto trt_in2 = at::clone(in2);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in, trt_in1, trt_in2});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converter, ATenIndexPutMultiDimensionConvertCorrectly) {
  const auto graph = R"IR(
    graph(%b.1 : Tensor,
          %a.1 : Tensor,
          %idx1.1 : Tensor,
          %idx2.1 : Tensor,
          %idx3.1 : Tensor,
          %idx4.1 : Tensor):
          %7 : int = prim::Constant[value=4]()
          %8 : bool = prim::Constant[value=0]()
          %9 : NoneType = prim::Constant()
          %10 : bool = prim::Constant[value=1]()
          %idx10.1 : Tensor = aten::to(%idx1.1, %7, %8, %8, %9)
          %idx20.1 : Tensor = aten::to(%idx2.1, %7, %8, %8, %9)
          %idx30.1 : Tensor = aten::to(%idx3.1, %7, %8, %8, %9)
          %idx40.1 : Tensor = aten::to(%idx4.1, %7, %8, %8, %9)
          %15 : Tensor?[] = prim::ListConstruct(%idx10.1, %idx20.1, %idx30.1, %idx40.1)
          %16 : Tensor = aten::index_put(%b.1, %15, %a.1, %10)
          return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto base = torch::zeros({3, 3, 3, 3}, {at::kCUDA}).to(torch::kFloat);
  auto update = torch::arange(1, 7, {at::kCUDA}).reshape({1, 6}).to(torch::kFloat);

  std::vector<int> in0;
  std::vector<int> in2;
  std::vector<int> in3;

  for (int i = 0; i < 2; i++) {
    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        in0.push_back(i);
        in2.push_back(n);
        in3.push_back(m);
      }
    }
  }

  auto index0 = torch::tensor(in0, {at::kCUDA}).reshape({3, 6}).to(torch::kInt32);
  auto index1 = torch::tensor({1, 1, 1}, {at::kCUDA}).reshape({3, 1}).to(torch::kInt32);
  auto index2 = torch::tensor(in2, {at::kCUDA}).reshape({3, 6}).to(torch::kInt32);
  auto index3 = torch::tensor(in3, {at::kCUDA}).reshape({3, 6}).to(torch::kInt32);

  auto jit_base = at::clone(base);
  auto jit_update = at::clone(update);
  auto jit_index0 = at::clone(index0);
  auto jit_index1 = at::clone(index1);
  auto jit_index2 = at::clone(index2);
  auto jit_index3 = at::clone(index3);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(
      g, params, {jit_base, jit_update, jit_index0, jit_index1, jit_index2, jit_index3});

  auto trt_base = at::clone(base);
  auto trt_update = at::clone(update);
  auto trt_index0 = at::clone(index0);
  auto trt_index1 = at::clone(index1);
  auto trt_index2 = at::clone(index2);
  auto trt_index3 = at::clone(index3);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(
      g, params, {trt_base, trt_update, trt_index0, trt_index1, trt_index2, trt_index3});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converter, ATenIndexPutPartDimensionConvertCorrectly) {
  const auto graph = R"IR(
    graph(%y.1 : Tensor,
          %mask.1 : Tensor):
          %3 : int = prim::Constant[value=4]()
          %4 : bool = prim::Constant[value=0]()
          %5 : NoneType = prim::Constant()
          %6 : bool = prim::Constant[value=1]()
          %mask0.1 : Tensor = aten::to(%mask.1, %3, %4, %4, %5)
          %8 : Tensor?[] = prim::ListConstruct(%mask0.1)
          %z.1 : Tensor = aten::index(%y.1, %8)
          %10 : Tensor = aten::index_put(%y.1, %8, %z.1, %6)
          return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 2, {10, 10}, {at::kCUDA}).to(at::kFloat);
  auto in1 = torch::arange(0, 6, {at::kCUDA}).to(torch::kInt32);

  auto jit_in = at::clone(in);
  auto jit_in1 = at::clone(in1);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in, jit_in1});

  auto trt_in = at::clone(in);
  auto trt_in1 = at::clone(in1);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in, trt_in1});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converter, ATenIndexPutPartDimensionExpandedConvertCorrectly) {
  const auto graph = R"IR(
    graph(%b.1 : Tensor,
          %a.1 : Tensor,
          %idx1.1 : Tensor,
          %idx2.1 : Tensor,
          %idx3.1 : Tensor):
          %7 : int = prim::Constant[value=4]()
          %8 : bool = prim::Constant[value=0]()
          %9 : NoneType = prim::Constant()
          %10 : bool = prim::Constant[value=1]()
          %idx10.1 : Tensor = aten::to(%idx1.1, %7, %8, %8, %9)
          %idx20.1 : Tensor = aten::to(%idx2.1, %7, %8, %8, %9)
          %idx30.1 : Tensor = aten::to(%idx3.1, %7, %8, %8, %9)
          %15 : Tensor?[] = prim::ListConstruct(%idx10.1, %idx20.1, %idx30.1)
          %16 : Tensor = aten::index_put(%b.1, %15, %a.1, %10)
          return (%16))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto base = torch::zeros({4, 3, 2}, {at::kCUDA}).to(torch::kFloat);
  auto update = torch::tensor({-1, -2}, {at::kCUDA}).to(torch::kFloat);

  auto index0 = torch::tensor({0}, {at::kCUDA}).to(torch::kInt32);
  auto index1 = torch::tensor({0, 1, 2}, {at::kCUDA}).reshape({3, 1}).to(torch::kInt32);
  auto index2 = torch::tensor({0, 1}, {at::kCUDA}).reshape({1, 2}).to(torch::kInt32);

  auto jit_base = at::clone(base);
  auto jit_update = at::clone(update);
  auto jit_index0 = at::clone(index0);
  auto jit_index1 = at::clone(index1);
  auto jit_index2 = at::clone(index2);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results =
      torch_tensorrt::tests::util::RunGraph(g, params, {jit_base, jit_update, jit_index0, jit_index1, jit_index2});

  auto trt_base = at::clone(base);
  auto trt_update = at::clone(update);
  auto trt_index0 = at::clone(index0);
  auto trt_index1 = at::clone(index1);
  auto trt_index2 = at::clone(index2);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(
      g, params, {trt_base, trt_update, trt_index0, trt_index1, trt_index2});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}