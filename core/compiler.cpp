#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include "NvInfer.h"

#include "ATen/core/function_schema.h"
#include "ATen/core/jit_type.h"

#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/ir_views.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/lower_graph.h"
#include "torch/csrc/jit/passes/pass_manager.h"
#include "torch/custom_class.h"

#include "core/compiler.h"

#include "core/conversion/conversion.h"
#include "core/lowering/lowering.h"
#include "core/partitioning/partitioning.h"
#include "core/runtime/runtime.h"

namespace trtorch {
namespace core {

void AddEngineToGraph(
    torch::jit::script::Module mod,
    std::shared_ptr<torch::jit::Graph>& g,
    const std::string& serialized_engine,
    std::string engine_id = "",
    bool fallback = false) {
  auto engine_ptr = c10::make_intrusive<runtime::TRTEngine>(mod._ivalue()->name() + engine_id, serialized_engine);
  // Get required metadata about the engine out
  auto num_io = engine_ptr->num_io;
  auto name = engine_ptr->name;

  //..
  // Add the engine as an attribute of the module, this will let the engine be
  // serialized and deserialized
  mod.register_attribute(
      name,
      c10::getCustomClassType<c10::intrusive_ptr<runtime::TRTEngine>>(),
      c10::IValue(std::move(engine_ptr)),
      false);

  // Add the module as an input into the graph
  auto self = g->addInput("self_1");
  self->setType(mod.type());

  // Start by retriveing the engine from the module attribute list
  auto engine_node = g->createGetAttr(self, name);
  g->block()->appendNode(engine_node);

  // Add inputs to the graph corresponding to the number of input tensors
  // expected by the engine Also store those inputs in a vector so that they can
  // be coalesced into a single list at runtime
  std::vector<torch::jit::Value*> engine_inputs;
  for (uint64_t i = 0; i < num_io.first; i++) {
    auto in_val = g->addInput(std::string("input_") + std::to_string(i));
    in_val->setType(c10::TensorType::get());
    engine_inputs.push_back(in_val);
  }

  // Create a node that will merge all of the input tensors into a single list
  // argument to the trt::execute_engine op Creates: prim::ListConstruct(<input
  // tensors>)
  auto input_list_node = g->createList(c10::TensorType::get(), torch::jit::ArrayRef<torch::jit::Value*>(engine_inputs));
  g->block()->appendNode(input_list_node);

  // Make a list of inputs to the actual trt::execute_engine op
  // Note: Ordering of list and then engine is because we can pop off the engine
  // first which contains all the metadata needed for execution
  std::vector<torch::jit::Value*> execute_node_inputs;
  execute_node_inputs.push_back(input_list_node->outputs()[0]);
  execute_node_inputs.push_back(engine_node->outputs()[0]);

  // Create the actual execution node trt::execute_engine using the assembled
  // inputs
  auto execute_node = g->create(
      c10::Symbol::fromQualString("tensorrt::execute_engine"),
      torch::jit::ArrayRef<torch::jit::Value*>(execute_node_inputs),
      1);
  g->block()->appendNode(execute_node);
  execute_node->outputs()[0]->setType(c10::ListType::ofTensors());

  // Create a node to unpack the list into seperate tensors, in the case of
  // there being only one tensor, the tensor will be returned, otherwise they
  // are returned as a tuple of tensors. Creates: prim::ListUnpack(<engine
  // output>)
  auto unpack_node = g->createListUnpack(execute_node->outputs()[0], num_io.second);
  g->block()->appendNode(unpack_node);

  // If there are multiple output tensors from TensorRT we wrap them in a tuple
  // to return, convert to tuple only when we only have 1 segmented graph
  if (!fallback && unpack_node->outputs().size() > 1) {
    // Creates prim::TupleConstruct(<output tensors>) using outputs of the
    // unpack node
    auto return_tuple_node = g->createTuple(unpack_node->outputs());
    g->block()->appendNode(return_tuple_node);
    // Set the output as the produced tuple
    g->registerOutput(return_tuple_node->outputs()[0]);
  } else {
    // if fallback is enabled, multiple outputs will be registered
    for (size_t i = 0; i < unpack_node->outputs().size(); ++i) {
      g->registerOutput(unpack_node->outputs()[i]);
    }
  }

  LOG_DEBUG(*g << "(AddEngineToGraph)\n");

  return;
}

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod, std::string method_name) {
  // Go through Lowering to simplify graph and extract weight parameters
  auto graph_and_parameters = lowering::Lower(mod, method_name);

  auto g = graph_and_parameters.first;
  LOG_DEBUG(*g << "(CheckMethodOperatorSupport)\n");

  return conversion::VerifyConverterSupportForBlock(g->block());
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod, std::string method_name, CompileSpec cfg) {
  // Go through Lowering to simplify graph and extract weight parameters
  auto graph_and_parameters = lowering::Lower(mod, method_name);

  auto convert_cfg = std::move(cfg.convert_info);
  auto g = graph_and_parameters.first;

  auto params = graph_and_parameters.second;
  auto named_params = conversion::get_named_params(g->inputs(), params);

  LOG_INFO(*g << "(CompileGraph)\n");

  auto engine = conversion::ConvertBlockToEngine(g->block(), convert_cfg, named_params);
  return std::move(engine);
}

void AddSegmentedBlockToGraph(
    std::shared_ptr<torch::jit::Graph>& g,
    partitioning::SegmentedBlock& seg,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new_g) {
  // old_to_new_g contains: original global graph value => new global graph value,
  // mini_to_new_g: mini graph value -> new graph value
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> mini_to_new_g;
  size_t input_idx = 0;
  if (seg.target() == partitioning::SegmentedBlock::kTensorRT && g->inputs().size() > 0) {
    if (g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
      auto self = g->insertInput(0, "self_1");
      self->setType(seg.inputs()[0]->type());
    }
    mini_to_new_g[seg.inputs()[input_idx++]] = g->inputs()[0];
  }

  for (auto& raw_input : seg.raw_inputs()) {
    if (old_to_new_g.count(raw_input)) {
      mini_to_new_g[seg.inputs()[input_idx++]] = old_to_new_g[raw_input];
    }
  }

  for (const auto n : seg.nodes()) {
    util::cloneNode(n, g, mini_to_new_g);
  }

  // original graph value => new global graph value
  for (size_t i = 0; i < seg.raw_outputs().size(); ++i) {
    old_to_new_g[seg.raw_outputs()[i]] = mini_to_new_g[seg.outputs()[i]];
  }
  size_t offset = seg.target() == partitioning::SegmentedBlock::kTensorRT ? 1 : 0;
  for (size_t i = 0; i < seg.raw_inputs().size(); ++i) {
    if (!old_to_new_g.count(seg.raw_inputs()[i])) {
      old_to_new_g[seg.raw_inputs()[i]] = mini_to_new_g[seg.inputs()[i + offset]];
    }
  }

  return;
}

typedef std::pair<std::shared_ptr<torch::jit::Graph>, std::unordered_map<torch::jit::Value*, torch::jit::Value*>>
    GraphAndMapping;

void AddIfBlockToGraph(
    std::shared_ptr<torch::jit::Graph>& new_g,
    torch::jit::Node* if_node,
    const std::vector<GraphAndMapping>& graph_and_mappings,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new_g) {
  torch::jit::IfView if_view(if_node);

  // create a new if node in new_g and add corresponding inputs
  auto new_if = new_g->insertNode(new_g->create(torch::jit::prim::If, {}, 0));
  new_if->addInput(util::getOrAddInputForValue(if_view.cond(), new_g, old_to_new_g));

  // iterate over all blocks and add them to new created prim::If
  for (auto graph_and_mapping : graph_and_mappings) {
    auto new_if_block = new_if->addBlock();
    auto cur_block_graph = graph_and_mapping.first;
    auto cur_block_mapping = graph_and_mapping.second;
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> block_graph_to_new_g;
    for (auto& i : cur_block_mapping) {
      // for every pair in then_mapping, old_value => mini graph value, if old_value also appears in old_to_new_g, then
      // it's mini graph's input
      if (old_to_new_g.count(i.first)) {
        block_graph_to_new_g[i.second] = old_to_new_g[i.first];
      }
    }

    auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, new_g, block_graph_to_new_g); };
    new_if_block->cloneFrom(cur_block_graph->block(), env);
    if (cur_block_graph->inputs()[0]->type()->str().find("__torch__") != std::string::npos) {
      if (new_g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
        auto self = new_g->insertInput(0, "self_1");
        self->setType(cur_block_graph->inputs()[0]->type());
      }
      block_graph_to_new_g[cur_block_graph->inputs()[0]] = new_g->inputs()[0];
    }
    for (int i = cur_block_graph->inputs().size() - 1; i >= 0; --i) {
      new_if_block->inputs()[i]->replaceAllUsesWith(block_graph_to_new_g[cur_block_graph->inputs()[i]]);
      new_if_block->eraseInput(i);
    }
  }
  for (auto ov : if_view.outputs()) {
    auto no = new_if->addOutput();
    old_to_new_g[ov] = no;
    no->copyMetadata(ov);
  }
  return;
}

GraphAndMapping ConstructFallbackGraph(
    torch::jit::script::Module& new_mod,
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue> input_ivalues_map,
    CompileSpec cfg,
    conversion::GraphParams named_params) {
  auto convert_cfg = cfg.convert_info;
  auto partition_info = cfg.partition_info;

  auto new_g = std::make_shared<torch::jit::Graph>();

  auto segmented_blocks = partitioning::Partition(block, input_ivalues_map, partition_info);

  // the mapping from lowering graph => fallback global graph
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_g;
  for (auto input : block->inputs()) {
    util::getOrAddInputForValue(input, new_g, old_to_new_g);
  }

  for (auto& seg_block : segmented_blocks) {
    LOG_INFO(*seg_block.g() << "(GraphInSegmentedBlock)\n");
    std::ostringstream trt_engine_id;
    trt_engine_id << reinterpret_cast<const int*>(&seg_block);

    if (seg_block.target() == partitioning::SegmentedBlock::kTensorRT) {
      std::vector<ir::InputRange> input_ranges;
      for (auto& shape : seg_block.in_shape()) {
        input_ranges.push_back(ir::InputRange(shape));
      }
      // update the input ranges for each segments
      convert_cfg.input_ranges = input_ranges;
      auto engine = conversion::ConvertBlockToEngine(seg_block.block(), convert_cfg, named_params);
      auto temp_g = std::make_shared<torch::jit::Graph>();
      AddEngineToGraph(new_mod, temp_g, engine, trt_engine_id.str(), true);

      seg_block.update_graph(temp_g);
      AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
    } else {
      if (seg_block.raw_nodes()[0]->kind() == torch::jit::prim::If) {
        auto if_node = seg_block.raw_nodes()[0];

        // convert the 2 blocks in prim::if and get the converted graph with mappings
        std::vector<GraphAndMapping> graph_and_mappings;
        for (auto cur_block : if_node->blocks()) {
          graph_and_mappings.push_back(
              ConstructFallbackGraph(new_mod, cur_block, input_ivalues_map, cfg, named_params));
        }
        AddIfBlockToGraph(new_g, if_node, graph_and_mappings, old_to_new_g);

      } else {
        AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
      }
    }
  }

  for (auto& output : block->outputs()) {
    if (old_to_new_g.count(output)) {
      new_g->registerOutput(old_to_new_g[output]);
    }
  }
  return {new_g, old_to_new_g};
}

torch::jit::script::Module CompileGraphWithFallback(const torch::jit::script::Module& mod, CompileSpec cfg) {
  // TODO: Should be doing a functional transform but need PR #31978
  // [jit] More robust mangling
  // torch::jit::script::Module new_mod = mod.clone();
  torch::jit::script::Module new_mod(mod._ivalue()->name() + "_trt");
  std::vector<std::shared_ptr<torch::jit::Graph>> graphs;
  for (const torch::jit::script::Method& method : mod.get_methods()) {
    // Don't convert hidden methods
    if (method.name().rfind("_", 0)) {
      auto new_g = std::make_shared<torch::jit::Graph>();
      auto graph_and_parameters = lowering::Lower(mod, method.name());

      auto g = graph_and_parameters.first;
      auto params = graph_and_parameters.second;
      auto named_params = conversion::get_named_params(g->inputs(), params);
      LOG_INFO(*g << "(LoweringGraph)\n");

      std::unordered_map<torch::jit::Value*, ir::InputRange> input_ranges;
      for (size_t i = 0; i < g->inputs().size(); ++i) {
        input_ranges.insert({g->inputs()[i], cfg.convert_info.input_ranges[i]});
      }
      auto input_ivalues_map = partitioning::generateRandomInputs(input_ranges);
      auto graph_and_mapping = ConstructFallbackGraph(new_mod, g->block(), input_ivalues_map, cfg, named_params);
      new_g = graph_and_mapping.first;
      LOG_INFO(*new_g << "(FallbackGraph)\n");

      // if there is no tensorrt engine self in fallback graph, there is no conversion, we just return the initial
      // module
      if (new_g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
        LOG_WARNING("Didn't generate any TensorRT engines, the compiler did nothing\n");
        return mod;
      }

      auto new_method = new_mod._ivalue()->compilation_unit()->create_function(method.name(), new_g);
      auto schema = util::GenerateGraphSchema(new_method->name(), new_g);
      new_mod.type()->addMethod(new_method);
      new_method->setSchema(schema);
    }
  }

  return new_mod;
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& mod, CompileSpec cfg) {
  // TODO: not sure how to deal with duplicated code here, so just cut out a branch temporally
  if (cfg.partition_info.enabled) {
    return CompileGraphWithFallback(mod, cfg);
  }
  // TODO: Should be doing a functional transform but need PR #31978
  // [jit] More robust mangling
  // torch::jit::script::Module new_mod = mod.clone();
  torch::jit::script::Module new_mod(mod._ivalue()->name() + "_trt");
  std::vector<std::shared_ptr<torch::jit::Graph>> graphs;
  for (const torch::jit::script::Method& method : mod.get_methods()) {
    // Don't convert hidden methods
    if (method.name().rfind("_", 0)) {
      auto engine = ConvertGraphToTRTEngine(mod, method.name(), cfg);
      auto new_g = std::make_shared<torch::jit::Graph>();
      AddEngineToGraph(new_mod, new_g, engine);
      auto new_method = new_mod._ivalue()->compilation_unit()->create_function(method.name(), new_g);
      auto schema = util::GenerateGraphSchema(new_method->name(), new_g);
      new_mod.type()->addMethod(new_method);
      new_method->setSchema(schema);
    }
  }

  return new_mod;
}

torch::jit::script::Module EmbedEngineInNewModule(const std::string& engine) {
  std::ostringstream engine_id;
  engine_id << reinterpret_cast<const int*>(&engine);
  torch::jit::script::Module new_mod("tensorrt_engine_mod_" + engine_id.str());
  auto new_g = std::make_shared<torch::jit::Graph>();
  AddEngineToGraph(new_mod, new_g, engine);
  auto new_method = new_mod._ivalue()->compilation_unit()->create_function("forward", new_g);
  auto schema = util::GenerateGraphSchema(new_method->name(), new_g);
  new_mod.type()->addMethod(new_method);
  new_method->setSchema(schema);

  return new_mod;
}

void set_device(const int gpu_id) {
  TRTORCH_ASSERT(cudaSetDevice(gpu_id) == cudaSuccess, "Unable to set CUDA device: " << gpu_id);
}

} // namespace core
} // namespace trtorch
