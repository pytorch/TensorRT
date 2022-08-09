#include <string>
#include "core/compiler.h"
#include "core/util/trt_util.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/script.h"

bool checkSegmentedBlockInputType(
    const torch_tensorrt::core::partitioning::SegmentedBlock& segmented_block,
    const std::function<bool(torch::jit::TypePtr)>& condition) {
  for (auto input : segmented_block.raw_inputs()) {
    if (!condition(input->type())) {
      return false;
    }
  }
  return true;
}

int count_trt_engines(std::shared_ptr<torch::jit::Graph> g) {
  int count = 0;
  for (const auto n : g->nodes()) {
    if (n->kind().toQualString() == std::string("tensorrt::execute_engine")) {
      ++count;
    }
  }
  return count;
}

TEST(Partitioning, ResolveNonTensorInputsForIFBlockCorrectly) {
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

TEST(Partitioning, ResolveNonTensorInputsCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 3, 3, 3, strides=[27, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=0]()
            %8 : int = aten::size(%0, %7)
            %9 : Tensor = aten::log_sigmoid(%0)
            %10 : Tensor = aten::_convolution(%9, %1, %2, %4, %3, %4, %5, %3, %8, %5, %5, %6, %6)
            %11 : Tensor = aten::relu(%10)
            return (%11))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 3, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));

  torch_tensorrt::core::ir::CollectionInputSpecMap inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], {inputs[i]}});
    input_types.insert({g->inputs()[i], {{at::kFloat}}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info, fallback_nodes);

  int torch_block_cnt = 0, trt_block_cnt = 0;
  for (const auto& segmented_block : segmented_blocks) {
    if (segmented_block.target() == torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT) {
      ++trt_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(segmented_block, [](torch::jit::TypePtr type_ptr) {
        return type_ptr->isSubtypeOf(torch::jit::TensorType::get());
      }));
    } else {
      ++torch_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(segmented_block, [](torch::jit::TypePtr type_ptr) {
        return type_ptr->isSubtypeOf(torch::jit::TensorType::get()) ||
            type_ptr->isSubtypeOf(torch::jit::ListType::ofTensors());
      }));
    }
  }
  ASSERT_TRUE(trt_block_cnt == 1 && torch_block_cnt == 1);
}

TEST(Partitioning, ResolveTensorListInputsInTrtCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 6, 3, 3, strides=[54, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=1]()
            %8 : int = prim::Constant[value=0]()
            %9 : Tensor[] = prim::ListConstruct(%0, %0)
            %10 : Tensor = aten::cat(%9, %8)
            %11 : Tensor = aten::log_sigmoid(%10)
            %12 : Tensor = aten::cat(%9, %7)
            %13 : Tensor = aten::_convolution(%12, %1, %2, %4, %3, %4, %5, %3, %7, %5, %5, %6, %6)
            %14 : Tensor = aten::relu(%13)
            %15 : (Tensor, Tensor) = prim::TupleConstruct(%11, %14)
            return (%15))IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16, 6, 3, 3}));
  inputs.push_back(torch_tensorrt::core::ir::Input({16}));

  std::unordered_map<const torch::jit::Value*, std::vector<torch_tensorrt::core::ir::Input>> inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], {inputs[i]}});
    input_types.insert({g->inputs()[i], {{at::kFloat}}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  std::vector<torch_tensorrt::core::partitioning::SegmentedBlock> segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info, fallback_nodes);

  int torch_block_cnt = 0, trt_block_cnt = 0;
  for (const auto& segmented_block : segmented_blocks) {
    if (segmented_block.target() == torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT) {
      ++trt_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(segmented_block, [](torch::jit::TypePtr type_ptr) {
        return type_ptr->isSubtypeOf(torch::jit::TensorType::get());
      }));
    } else {
      ++torch_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(segmented_block, [](torch::jit::TypePtr type_ptr) {
        return type_ptr->isSubtypeOf(torch::jit::TensorType::get()) ||
            type_ptr->isSubtypeOf(torch::jit::ListType::ofTensors());
      }));
    }
  }
  ASSERT_TRUE(trt_block_cnt == 2 && torch_block_cnt == 2);
}

TEST(Partitioning, ConvertForTensorListInputsInFallbackCorrectly) {
  const auto graph = R"IR(
          graph(%0 : Float(1, 3, 16, 16, strides=[768, 256, 16, 1]),
                %1 : Float(16, 6, 3, 3, strides=[54, 9, 3, 1]),
                %2 : Float(16, strides=[1])):
            %3 : int[] = prim::Constant[value=[0, 0]]()
            %4 : int[] = prim::Constant[value=[1, 1]]()
            %5 : bool = prim::Constant[value=0]()
            %6 : bool = prim::Constant[value=1]()
            %7 : int = prim::Constant[value=1]()
            %8 : int = prim::Constant[value=0]()
            %9 : Tensor[] = prim::ListConstruct(%0, %0)
            %11 : Tensor = aten::log_sigmoid(%0)
            %12 : Tensor = aten::cat(%9, %7)
            %13 : Tensor = aten::_convolution(%12, %1, %2, %4, %3, %4, %5, %3, %7, %5, %5, %6, %6)
            %14 : Tensor = aten::relu(%13)
            %15 : (Tensor, Tensor) = prim::TupleConstruct(%11, %14)
            return (%15))IR";
  auto parsed_g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, parsed_g.get());

  auto g = std::make_shared<torch::jit::Graph>();
  std::vector<std::vector<int64_t>> all_shapes{{16, 6, 3, 3}, {16}};
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> tensor_to_constant;
  for (size_t i = 0; i < all_shapes.size(); ++i) {
    auto in = at::randn(all_shapes[i], {at::kCUDA});
    torch::jit::IValue cur_val = in.clone();
    auto new_val = g->insertConstant(cur_val);
    tensor_to_constant[parsed_g->inputs()[i + 1]] = new_val;
  }
  for (auto node : parsed_g->nodes()) {
    if (node->kind() == torch::jit::prim::Constant)
      continue;
    torch_tensorrt::core::util::cloneNode(node, g, tensor_to_constant);
  }
  g->registerOutput(tensor_to_constant[parsed_g->outputs()[0]]);

  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({1, 3, 16, 16}));
  torch_tensorrt::core::CompileSpec cfg(inputs);
  cfg.partition_info.enabled = true;
  torch::jit::script::Module mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(mod.type());
  auto cur_method = mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema = torch_tensorrt::core::util::GenerateGraphSchema(cur_method->name(), g);
  mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  torch::jit::script::Module new_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
  auto fallback_g = new_mod.get_method("forward").graph();
  int count = count_trt_engines(fallback_g);
  ASSERT_TRUE(count == 1);
}

TEST(Partitioning, ResolveOnlyNeccessaryNonTensorInputs) {
  /* parseIR does not support "= aten::_set_item" so we will build this graph manually
    const auto graph = R"IR(
    graph(%x : Tensor,
      %y : Tensor):
    %2 : str = prim::Constant[value="INS"]()
    %3 : str = prim::Constant[value="OUTS"]()
    %4 : bool = prim::Constant[value=0]()
    %5 : int = prim::Constant[value=-1]()
    %6 : Dict(str, Tensor) = prim::DictConstruct()
     = aten::_set_item(%6, %2, %x)
    %7 : Tensor = aten::__getitem__(%6, %2)
    %8 : Tensor = aten::lt(%7, %y)
    %9 : Tensor?[] = prim::ListConstruct(%8)
    %10 : int = prim::dtype(%7)
    %11 : Device = prim::device(%7)
    %12 : Tensor = aten::tensor(%5, %10, %11, %4)
    %13 : Tensor = aten::index_put_(%7, %9, %12, %4)
     = aten::_set_item(%6, %3, %7)
    %14 : Tensor = aten::__getitem__(%6, %2)
    %15 : Tensor = aten::__getitem__(%6, %3)
    return (%14, %15))IR";
  */
  auto g = std::make_shared<torch::jit::Graph>();
  auto x = g->insertInput(0, "x");
  auto y = g->insertInput(1, "y");
  torch::jit::IValue ins_key("INS");
  auto ins_key_val = g->insertConstant(ins_key);
  torch::jit::IValue outs_key("OUTS");
  auto outs_key_val = g->insertConstant(outs_key);
  torch::jit::IValue zero(0);
  auto false_const_val = g->insertConstant(zero);
  false_const_val->setType(c10::BoolType::get());
  torch::jit::IValue neg_one(-1);
  auto neg_one_const_val = g->insertConstant(neg_one);
  auto dict_node = g->createDict(
      ins_key_val->type(),
      x->type(),
      torch::jit::ArrayRef<torch::jit::Value*>(),
      torch::jit::ArrayRef<torch::jit::Value*>());
  g->insertNode(dict_node);
  auto set_node = g->create(
      torch::jit::Symbol::fromQualString("aten::_set_item"),
      torch::jit::ArrayRef<torch::jit::Value*>{dict_node->output(), ins_key_val, x},
      0);
  g->insertNode(set_node);
  auto get_node = g->create(
      torch::jit::Symbol::fromQualString("aten::__getitem__"),
      torch::jit::ArrayRef<torch::jit::Value*>{dict_node->output(), ins_key_val},
      1);
  g->insertNode(get_node);
  auto lt_node = g->create(
      torch::jit::Symbol::fromQualString("aten::lt"),
      torch::jit::ArrayRef<torch::jit::Value*>{get_node->output(), y},
      1);
  g->insertNode(lt_node);
  auto list_node = g->createList(
      at::OptionalType::create(lt_node->output()->type()), torch::jit::ArrayRef<torch::jit::Value*>{lt_node->output()});
  g->insertNode(list_node);
  auto dtype_node = g->create(
      torch::jit::Symbol::fromQualString("prim::dtype"),
      torch::jit::ArrayRef<torch::jit::Value*>{get_node->output()},
      1);
  dtype_node->output()->setType(neg_one_const_val->type());
  g->insertNode(dtype_node);
  auto device_node = g->create(
      torch::jit::Symbol::fromQualString("prim::device"),
      torch::jit::ArrayRef<torch::jit::Value*>{get_node->output()},
      1);
  device_node->output()->setType(c10::DeviceObjType::get());
  g->insertNode(device_node);
  auto tensor_node = g->create(
      torch::jit::Symbol::fromQualString("aten::tensor"),
      torch::jit::ArrayRef<torch::jit::Value*>{
          neg_one_const_val, dtype_node->output(), device_node->output(), false_const_val},
      1);
  g->insertNode(tensor_node);
  auto index_put_node = g->create(
      torch::jit::Symbol::fromQualString("aten::index_put_"),
      torch::jit::ArrayRef<torch::jit::Value*>{
          get_node->output(), list_node->output(), tensor_node->output(), false_const_val},
      1);
  g->insertNode(index_put_node);
  auto out_set_node = g->create(
      torch::jit::Symbol::fromQualString("aten::_set_item"),
      torch::jit::ArrayRef<torch::jit::Value*>{dict_node->output(), outs_key_val, get_node->output()},
      0);
  g->insertNode(out_set_node);
  auto get_ins_node = g->create(
      torch::jit::Symbol::fromQualString("aten::__getitem__"),
      torch::jit::ArrayRef<torch::jit::Value*>{dict_node->output(), ins_key_val},
      1);
  g->insertNode(get_ins_node);
  auto get_outs_node = g->create(
      torch::jit::Symbol::fromQualString("aten::__getitem__"),
      torch::jit::ArrayRef<torch::jit::Value*>{dict_node->output(), outs_key_val},
      1);
  g->insertNode(get_outs_node);
  g->registerOutput(get_ins_node->output());
  g->registerOutput(get_outs_node->output());

  torch_tensorrt::core::partitioning::PartitionInfo partition_info;
  partition_info.enabled = true;
  std::vector<torch_tensorrt::core::ir::Input> inputs;
  inputs.push_back(torch_tensorrt::core::ir::Input({4, 4}));
  inputs.push_back(torch_tensorrt::core::ir::Input({4, 4}));

  torch_tensorrt::core::ir::CollectionInputSpecMap inputs_map;
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types;
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    inputs_map.insert({g->inputs()[i], {inputs[i]}});
    input_types.insert({g->inputs()[i], {{at::kFloat}}});
  }
  auto input_ivalues_map = torch_tensorrt::core::partitioning::generateRandomInputs(inputs_map, input_types);
  std::unordered_map<torch::jit::Node*, int> fallback_nodes;
  auto segmented_blocks =
      torch_tensorrt::core::partitioning::Partition(g->block(), input_ivalues_map, partition_info, fallback_nodes);

  int torch_block_cnt = 0, trt_block_cnt = 0;
  for (const auto& segmented_block : segmented_blocks) {
    if (segmented_block.target() == torch_tensorrt::core::partitioning::SegmentedBlock::kTensorRT) {
      ++trt_block_cnt;
      ASSERT_TRUE(checkSegmentedBlockInputType(segmented_block, [](torch::jit::TypePtr type_ptr) {
        return type_ptr->isSubtypeOf(torch::jit::TensorType::get());
      }));
    } else {
      ++torch_block_cnt;
      bool output_dict = false;
      bool input_dict = false;
      auto dict_type = dict_node->output()->type();
      for (auto in : segmented_block.raw_inputs()) {
        if (in->type()->isSubtypeOf(dict_type)) {
          input_dict = true;
        }
      }
      for (auto out : segmented_block.raw_outputs()) {
        if (out->type()->isSubtypeOf(dict_type)) {
          output_dict = true;
        }
      }
      EXPECT_TRUE(output_dict ^ input_dict);
    }
  }
  ASSERT_TRUE(trt_block_cnt == 1 && torch_block_cnt == 2);
}
