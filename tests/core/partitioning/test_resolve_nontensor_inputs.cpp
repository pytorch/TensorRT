#include <string>
#include "core/compiler.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

TEST(Partitioning, ResolveNonTensorInputsCorrectly) {
  const auto graph = R"IR(
        graph(%x : Tensor, %y : Tensor):
          %0 : int = prim::Constant[value=0]()
          %1 : int = prim::Constant[value=1]()
          %a : Tensor = aten::add(%x, %y, %1)
          %s : int = aten::size(%a, %1)
          %D3.1 : Tensor = prim::NumToTensor(%s)
          %19 : bool = aten::is_floating_point(%D3.1)
          %2 : Tensor = prim::If(%19)
            block0():
                %2.1 : Tensor = aten::sub(%a, %y, %1)
                -> (%2.1)
            block1():
                %2.2 : Tensor = aten::sub(%a, %y, %0)
                -> (%2.2)
          %3 : Tensor = prim::If(%19)
            block0():
                %3.1 : Tensor = aten::sub(%a, %y, %1)
                -> (%3.1)
            block1():
                %3.2 : Tensor = aten::sub(%a, %y, %0)
                -> (%3.2)
          %4 : Tensor = aten::add(%2, %3, %1)
          return (%4))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 4}));
  inputs.push_back(torch_tensorrt::core::ir::Input({3, 4}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  cfg.partition_info.forced_fallback_operators.push_back("aten::sub");
  cfg.convert_info.engine_settings.truncate_long_and_double = true;
  cfg.partition_info.truncate_long_and_double = true;

  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);

  auto in0 = at::randint(5, {3, 4}, {at::kCUDA});
  auto in1 = at::randint(5, {3, 4}, {at::kCUDA});

  auto jit_in0 = at::clone(in0);
  auto jit_in1 = at::clone(in1);
  auto trt_in0 = at::clone(in0);
  auto trt_in1 = at::clone(in1);

  auto jit_results = mod.forward({jit_in0, jit_in1});
  auto trt_results = new_mod.forward({trt_in0, trt_in1});

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results.toTensor(), trt_results.toTensor(), 2e-6));
}
