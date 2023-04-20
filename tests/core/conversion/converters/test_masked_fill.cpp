#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenMaskedFillZerosConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
      %44 : Device = prim::Constant[value="cuda"]()
      %8 : bool = prim::Constant[value=0]()
      %7 : None = prim::Constant()
      %f32_dtype: int = prim::Constant[value=11]()
      %1 : int = prim::Constant[value=0]() # bert.py:5:26
      %2 : int = prim::Constant[value=1]() # bert.py:5:32
      %33 : int = prim::Constant[value=2]() # bert.py:6:31
      %3 : int[] = prim::ListConstruct(%1, %1, %2)
      %4 : int[] = prim::ListConstruct(%2, %2, %1)
      %5 : int[][] = prim::ListConstruct(%3, %4)
      %9 : Tensor = aten::tensor(%5, %f32_dtype, %7, %8) # bert.py:5:11
      %mask.1 : Tensor = aten::to(%9, %44, %7, %8, %8) # bert.py:5:11
      %mask.2 : Tensor = trt::const(%mask.1)
      %34 : Tensor = aten::masked_fill(%x.1, %mask.1, %33) # bert.py:6:11
      return (%34, %mask.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::zeros({1, 2, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  torch_tensorrt::core::lowering::passes::RemoveNOPs(g);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}

TEST(Converters, ATenMaskedFillMixedTypesFloatIntConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor, %x.2 : Tensor):
      %val : float = prim::Constant[value=4.0]()
      %out : Tensor = aten::masked_fill(%x.1, %x.2, %val)
      return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  // Input is a float tensor, filled with an int --> expecting float tensor out
  auto in1 = at::rand({2, 3, 5, 7}, {at::kCUDA}).to(torch::kFloat32);
  auto in2 = (2 * at::rand({2, 3, 5, 7}, {at::kCUDA})).to(torch::kBool);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, in2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, in2});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));

  // Ensure data types match in outputs
  ASSERT_TRUE(jit_results[0].dtype() == trt_results[0].dtype());
}

TEST(Converters, ATenMaskedFillMixedTypesIntFloatConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor, %x.2 : Tensor):
      %val : int = prim::Constant[value=4]()
      %out : Tensor = aten::masked_fill(%x.1, %x.2, %val)
      return (%out))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  // Input is an integer tensor, filled with a float --> expecting integer tensor out
  auto in1 = at::rand({1, 3, 5, 7}, {at::kCUDA}).to(torch::kInt32);
  auto in2 = (2 * at::rand({1, 3, 5, 7}, {at::kCUDA})).to(torch::kBool);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, in2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, in2});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));

  // Ensure data types match in outputs
  ASSERT_TRUE(jit_results[0].dtype() == trt_results[0].dtype());
}