#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

TEST(LoweringPasses, ViewToReshapeCorrectly) {
  std::string source_graph = R"IR(
    graph(%x : Tensor, %1, %1.1):
        %0 : int = prim::Constant[value=0]()
        %2 : Tensor = aten::permute(%x, %1)
        %3 : Tensor = aten::contiguous(%2, %0)
        %4 : Tensor = aten::view(%3, %1.1)
        return (%4))IR";
  std::string target_graph = R"IR(
    graph(%x : Tensor, %1, %1.1):
        %0 : int = prim::Constant[value=0]()
        %2 : Tensor = aten::permute(%x, %1)
        %4 : Tensor = aten::reshape(%2, %1.1)
        return (%4))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kGRAPH);
  auto sg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(source_graph, &*sg);
  torch_tensorrt::core::lowering::passes::RemoveContiguous(sg);
  torch_tensorrt::core::lowering::passes::ViewToReshape(sg);

  auto tg = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(target_graph, &*tg);

  ASSERT_TRUE(!torch::jit::findPatternMatches(*tg, *sg).empty());
}

TEST(LoweringPasses, ViewToReshapeResultsCorrectly) {
  std::string graph = R"IR(
    graph(%x.1 : Tensor):
        %1 : int = prim::Constant[value=0]()
        %2 : int[] = prim::Constant[value=[1, 0, 3, 2]]()
        %5 : int = prim::Constant[value=0]()
        %p.1 : Tensor = aten::permute(%x.1, %2)
        %28 : int = prim::Constant[value=1]()
        %29 : int = prim::Constant[value=2]()
        %30 : int = prim::Constant[value=3]()
        %31 : int = prim::Constant[value=-1]()
        %size.1 : int[] = aten::size(%x.1)
        %33 : int = aten::__getitem__(%size.1, %5)
        %34 : int = aten::__getitem__(%size.1, %28)
        %35 : int = aten::mul(%33, %34)
        %36 : int = aten::__getitem__(%size.1, %29)
        %37 : int = aten::__getitem__(%size.1, %30)
        %38 : int = aten::mul(%36, %37)
        %39 : int[] = prim::ListConstruct(%31, %35, %38)
        %c : Tensor = aten::contiguous(%p.1, %1)
        %v.1 : Tensor = aten::view(%c, %39)
        return (%v.1))IR";

  torch_tensorrt::core::util::logging::get_logger().set_reportable_log_level(
      torch_tensorrt::core::util::logging::LogLevel::kDEBUG);

  auto parsed_g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, parsed_g.get());

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({2, 3, 4, 5}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("aten::permute");

  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = parsed_g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), parsed_g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), parsed_g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);

  auto in = at::randint(5, {2, 3, 4, 5}, {at::kCUDA});

  auto jit_in = at::clone(in);
  auto trt_in = at::clone(in);

  auto jit_results = mod.forward({jit_in});
  auto trt_results = new_mod.forward({trt_in});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results.toTensor(), trt_results.toTensor(), 2e-6));
}
