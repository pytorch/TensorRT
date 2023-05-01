#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, AtenPadConstantCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="constant"]()
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %2 : Scalar = prim::Constant[value=0.0]()
        %3 : Tensor = aten::constant_pad_nd(%0, %1, %2)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadConstantNoneValueCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="constant"]()
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : NoneType = prim::Constant()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %2 : Scalar = prim::Constant[value=0.0]()
        %3 : Tensor = aten::constant_pad_nd(%0, %1, %2)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadReflect1dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="reflect"]()
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %3 : Tensor = aten::reflection_pad1d(%0, %1)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadReflect2dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="reflect"]()
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : Tensor = aten::reflection_pad2d(%0, %1)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadReplicate1dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="replicate"]()
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3]]()
        %3 : Tensor = aten::replication_pad1d(%0, %1)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadReplicate2dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="replicate"]()
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3]]()
        %3 : Tensor = aten::replication_pad2d(%0, %1)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4, 5}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}

TEST(LoweringPasses, AtenPadReplicate3dCorrectly) {
  const auto source_graph = R"IR(
      graph(%0 : Tensor):
        %2 : str = prim::Constant[value="replicate"]()
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3, 1, 4]]()
        %3 : float = prim::Constant[value=0.0]()
        %4 : Tensor = aten::pad(%0, %1, %2, %3)
        return (%4))IR";

  const auto target_graph = R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[2, 3, 2, 3, 1, 4]]()
        %3 : Tensor = aten::replication_pad3d(%0, %1)
        return (%3))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::ReplaceAtenPad(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  auto in = at::randint(1, 10, {1, 3, 4, 5, 3}, {at::kCUDA});

  auto trt_in = at::clone(in);
  auto params = torch_tensorrt::core::ir::get_static_params(sg->inputs(), {});
  auto trt_results_sg = torch_tensorrt::tests::util::RunGraphEngine(sg, params, {trt_in});

  params = torch_tensorrt::core::ir::get_static_params(tg->inputs(), {});
  auto trt_results_tg = torch_tensorrt::tests::util::RunGraphEngine(tg, params, {trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(trt_results_sg[0], trt_results_tg[0], 2e-6));
}
