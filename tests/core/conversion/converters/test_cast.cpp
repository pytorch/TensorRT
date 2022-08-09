#include <torch/torch.h>
#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenBoolToFP32DTypeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %3 : int = prim::Constant[value=6]()
            %4 : int = prim::Constant[value=2]()
            %5 : bool = prim::Constant[value=0]()
            %6 : None = prim::Constant()
            %7 : Tensor = aten::ge(%x.1, %4)
            %out.1 : Tensor = aten::to(%7, %3, %5, %5, %6)

            return (%out.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 4, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(jit_results[0].scalar_type() == trt.scalar_type());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenBoolToINT32DTypeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %3 : int = prim::Constant[value=3]()
            %4 : int = prim::Constant[value=2]()
            %5 : bool = prim::Constant[value=0]()
            %6 : None = prim::Constant()
            %7 : Tensor = aten::ge(%x.1, %4)
            %out.1 : Tensor = aten::to(%7, %3, %5, %5, %6)

            return (%out.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 4, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(jit_results[0].scalar_type() == trt.scalar_type());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenBoolToINT32DeviceDTypeConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor):
            %3 : int = prim::Constant[value=3]()
            %4 : int = prim::Constant[value=2]()
            %5 : bool = prim::Constant[value=0]()
            %6 : None = prim::Constant()
            %7 : Tensor = aten::ge(%x.1, %4)
            %44 : Device = prim::Constant[value="cuda"]()
            %out.1 : Tensor = aten::to(%7, %44, %3, %5, %5)
            return (%out.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 4, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(jit_results[0].scalar_type() == trt.scalar_type());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenBoolToINT32TensorConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%x.1 : Tensor, %y.1 : Tensor):
            %3 : int = prim::Constant[value=2]()
            %4 : int = prim::Constant[value=3]()
            %5 : bool = prim::Constant[value=0]()
            %6 : None = prim::Constant()
            %y0.1 : Tensor = aten::to(%y.1, %4, %5, %5, %6)
            %8 : Tensor = aten::ge(%x.1, %3)
            %out.1 : Tensor = aten::to(%8, %y0.1, %5, %5, %6)
            %10 : Tensor = aten::mul(%out.1, %y0.1)
            return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3, 4, 3}, {at::kCUDA});
  auto in2 = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto jit_in2 = at::clone(in2);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in, jit_in2});

  auto trt_in = at::clone(in);
  auto trt_in2 = at::clone(in2);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in, trt_in2});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(jit_results[0].scalar_type() == trt.scalar_type());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenToSingleConvertsCorrectly) {
  const auto graph = R"IR(
    graph(%y.1 : Tensor):
            %4 : int = prim::Constant[value=6]()
            %5 : bool = prim::Constant[value=0]()
            %6 : None = prim::Constant()
            %y0.1 : Tensor = aten::to(%y.1, %4, %5, %5, %6)
            return (%y0.1))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, &*g);

  auto in = at::randint(1, 10, {3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});

  auto trt = trt_results[0].reshape(jit_results[0].sizes());
  ASSERT_TRUE(jit_results[0].scalar_type() == trt.scalar_type());
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenTypeAsConvertsCorrectly) {
  const auto graph = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %a : int = prim::Constant[value=1]()
        %4 : Tensor = aten::add(%0, %2, %a)
        %5 : Tensor = aten::gt(%1, %a)
        %6 : Tensor = aten::type_as(%4, %5)
        return (%6, %5))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, &*g);

  auto in1 = at::randint(1, 3, {3, 4, 3}, {at::kCUDA}).to(at::kFloat);
  auto in2 = at::randint(1, 3, {3, 4, 3}, {at::kCUDA});
  // Lower aten::type_as to aten::to.other
  torch_tensorrt::core::lowering::passes::ReduceToOperation(g);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {in1, in2});

  params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {in1, in2});

  ASSERT_TRUE(jit_results[0].scalar_type() == trt_results[0].scalar_type());
  ASSERT_TRUE(trt_results[0].scalar_type() == trt_results[1].scalar_type());
  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_results[0], trt_results[0].reshape_as(jit_results[0]), 2e-6));
}
