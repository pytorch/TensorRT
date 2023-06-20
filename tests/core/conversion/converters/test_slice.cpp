#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(Converters, ATenSliceConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %3 : int = prim::Constant[value=2]()
              %4 : int = prim::Constant[value=4]()
              %5 : int = prim::Constant[value=1]()
              %6 : int = prim::Constant[value=0]()
              %7 : Tensor = aten::select(%x.1, %6, %6)
              %8 : Tensor = aten::select(%7, %6, %5)
              %9 : Tensor = aten::slice(%8, %6, %5, %4, %3)
              %10 : Tensor = aten::slice(%9, %5, %2, %2, %5)
              return (%10))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {1, 3, 5, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceNegStartIndexConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int = prim::Constant[value=1]()
              %3 : int = prim::Constant[value=9223372036854775807]()
              %4 : int = prim::Constant[value=-2]()
              %5 : int = prim::Constant[value=0]()
              %6 : Tensor = aten::slice(%x.1, %5, %4, %3, %2)
              %7 : Tensor = aten::slice(%6, %2, %5, %3, %2)
              return (%7))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {6, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceNegEndIndexConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : int = prim::Constant[value=3]()
              %3 : int = prim::Constant[value=9223372036854775807]()
              %4 : int = prim::Constant[value=2]()
              %5 : int = prim::Constant[value=-3]()
              %6 : int = prim::Constant[value=1]()
              %7 : int = prim::Constant[value=-2]()
              %8 : int = prim::Constant[value=0]()
              %9 : Tensor = aten::slice(%x.1, %8, %8, %7, %6)
              %10 : Tensor = aten::slice(%9, %6, %8, %5, %6)
              %11 : Tensor = aten::slice(%10, %4, %8, %3, %6)
              %12 : Tensor = aten::slice(%11, %2, %8, %3, %6)
              return (%12))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {6, 5, 3, 3}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in});
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceListConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x : Tensor):
          %1 : NoneType = prim::Constant()
          %2 : int = prim::Constant[value=2]()
          %3 : int = prim::Constant[value=1]()
          %4 : int = prim::Constant[value=3]()
          %list : Tensor[] = aten::unbind(%x, %4)
          %slice : Tensor[] = aten::slice(%list, %1, %2, %3)
          %out.1 : Tensor, %out.2 : Tensor = prim::ListUnpack(%slice)
          return (%out.1, %out.2))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in_x = at::randint(1, 10, {6, 5, 3, 3}, {at::kCUDA});

  auto jit_in_x = at::clone(in_x);

  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in_x});

  auto trt_in_x = at::clone(in_x);
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngine(g, params, {trt_in_x});

  for (size_t i = 0; i < jit_results.size(); i++) {
    auto trt = trt_results[i].reshape(jit_results[i].sizes());
    ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[i], trt, 2e-6));
  }
}

TEST(Converters, ATenSliceDynamicBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=15]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicBatchLargeEndConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=9223372036854775807]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNegStartBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=-15]()
              %end : int = prim::Constant[value=15]()
              %step : int = prim::Constant[value=2]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNegEndBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=0]()
              %start : int = prim::Constant[value=1]()
              %end : int = prim::Constant[value=-2]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicNoneBatchConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %dim : int = prim::Constant[value=0]()
              %start : None = prim::Constant()
              %end : None = prim::Constant()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamicConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=1]()
              %start : int = prim::Constant[value=3]()
              %end : int = prim::Constant[value=32]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in dim 1, slice in dim 1
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, false);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}

TEST(Converters, ATenSliceDynamic2ConvertsCorrectly) {
  const auto graph = R"IR(
        graph(%x.1 : Tensor):
              %2 : None = prim::Constant()
              %dim : int = prim::Constant[value=1]()
              %start : int = prim::Constant[value=3]()
              %end : int = prim::Constant[value=17]()
              %step : int = prim::Constant[value=3]()
              %9 : Tensor = aten::slice(%x.1, %dim, %start, %end, %step)
              return (%9))IR";

  auto g = std::make_shared<torch::jit::Graph>();

  torch::jit::parseIR(graph, g.get());

  auto in = at::randint(1, 10, {16, 32}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(g->inputs(), {});
  auto jit_results = torch_tensorrt::tests::util::RunGraph(g, params, {jit_in});

  auto trt_in = at::clone(in);
  // dynamic shape in batch, slice in dim 1
  auto trt_results = torch_tensorrt::tests::util::RunGraphEngineDynamic(g, params, {trt_in}, true);
  auto trt = trt_results[0].reshape(jit_results[0].sizes());

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results[0], trt, 2e-6));
}