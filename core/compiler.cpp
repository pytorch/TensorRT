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
    int engine_id = 0,
    bool fallback = false) {
  auto engine_ptr =
      c10::make_intrusive<runtime::TRTEngine>(mod._ivalue()->name() + std::to_string(engine_id), serialized_engine);
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

  return;
}

typedef std::pair<std::shared_ptr<torch::jit::Graph>, std::unordered_map<torch::jit::Value*, torch::jit::Value*>>
    graph_with_mapping;

graph_with_mapping ConstructFallbackBlock(
    torch::jit::script::Module& new_mod,
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue> input_ivalues_map,
    CompileSpec cfg,
    int& trt_engine_id,
    conversion::GraphParams named_params) {
  auto convert_cfg = cfg.convert_info;
  auto partition_info = cfg.partition_info;

  auto new_g = std::make_shared<torch::jit::Graph>();

  auto segmented_blocks = partitioning::Partition(block, input_ivalues_map, partition_info);

  // the mapping from lowering graph => fallback global graph
  std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_g;
  for (auto input : block->inputs()) {
    //    if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
    util::getOrAddInputForValue(input, new_g, old_to_new_g);
    //    }
  }

  for (auto& seg_block : segmented_blocks) {
    LOG_INFO(*seg_block.g() << "(GraphInSegmentedBlock)\n");

    if (seg_block.target() == partitioning::SegmentedBlock::kTensorRT) {
      std::vector<ir::InputRange> input_ranges;
      for (auto& shape : seg_block.in_shape()) {
        input_ranges.push_back(ir::InputRange(shape));
      }
      // update the input ranges for each segments
      convert_cfg.input_ranges = input_ranges;
      auto engine = conversion::ConvertBlockToEngine(seg_block.block(), convert_cfg, named_params);
      auto temp_g = std::make_shared<torch::jit::Graph>();
      AddEngineToGraph(new_mod, temp_g, engine, trt_engine_id++, true);

      seg_block.update_graph(temp_g);
      AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
    } else {
      if (seg_block.raw_nodes()[0]->kind() == torch::jit::prim::Loop) {
        // the LoopView of the original loop node
        auto outer_node = seg_block.raw_nodes()[0];
        torch::jit::LoopView lv(outer_node);
        auto block_outputs = lv.bodyBlock()->outputs();
        auto block_inputs = lv.bodyBlock()->inputs();

        // get the required input shapes for the block in loop
        cfg.convert_info.input_ranges = seg_block.in_shape();
        std::unordered_map<torch::jit::Value*, ir::InputRange> cur_input_ranges;
        // for every loop, we have carriedInput in prim::Loop node, and bodyCarriedInputs in the block
        // Some values used in Loop block appears in bodyCarriedInputs, some are outer block values in
        // seg_block_raw_inputs
        for (size_t i = 0; i < seg_block.in_shape().size(); ++i) {
          cur_input_ranges.insert({seg_block.raw_inputs()[i], seg_block.in_shape()[i]});
        }
        for (size_t i = 0; i < lv.bodyCarriedInputs().size(); ++i) {
          cur_input_ranges.insert({lv.bodyCarriedInputs()[i], cur_input_ranges.at(lv.carriedInputs()[i])});
        }
        // we pass in torch::jit::IValue here in order to pass in iterators
        auto cur_input_ivalues_map = partitioning::generateRandomInputs(cur_input_ranges);

        torch::jit::Block* loop_block = seg_block.raw_nodes()[0]->blocks()[0];
        printf("current trip count: %s\n", lv.currentTripCount()->debugName().c_str());
        printf("current trip count out node: %s\n", lv.currentTripCount()->node()->kind().toQualString());
        cur_input_ivalues_map[loop_block->inputs()[0]] = torch::jit::IValue(0);
        //        input_ivalues_map.insert({lv.currentTripCount, torch::jit::IValue(0)});

        // get the converted hybrid graph in the loop
        auto loop_graph_mapping =
            ConstructFallbackBlock(new_mod, loop_block, cur_input_ivalues_map, cfg, trt_engine_id, named_params);
        auto loop_graph = loop_graph_mapping.first;
        auto loop_mapping = loop_graph_mapping.second;
        LOG_INFO(*loop_graph << "(LoopGraph)\n");

        // create a new loop for current graph
        auto new_loop =
            new_g->insertNode(new_g->create(torch::jit::prim::Loop, {}, 0))->setSourceRange(outer_node->sourceRange());

        // add inputs for new_loop node in new_g
        new_loop->addInput(util::getOrAddInputForValue(lv.maxTripCount(), new_g, old_to_new_g));
        new_loop->addInput(util::getOrAddInputForValue(lv.inputCond(), new_g, old_to_new_g));

        for (auto ci : lv.carriedInputs()) {
          new_loop->addInput(util::getOrAddInputForValue(ci, new_g, old_to_new_g));
        }

        // assure that new_g->inputs()[0] is fallback_trt, if not, insert it to new_g
        std::unordered_map<torch::jit::Value*, torch::jit::Value*> mini_to_new_g;
        if (new_g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
          auto self = new_g->insertInput(0, "self_1");
          self->setType(loop_graph->inputs()[0]->type());
        }
        mini_to_new_g[loop_graph->inputs()[0]] = new_g->inputs()[0];

        // add the block into our new loop
        auto new_loop_body = new_loop->addBlock();
        auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, new_g, mini_to_new_g); };
        new_loop_body->cloneFrom(loop_graph->block(), env);
        // replace the inputs()[0] in loop with currentTripCount();
        new_loop_body->inputs()[0]->replaceAllUsesWith(new_g->inputs()[0]);
        new_loop_body->inputs()[0]->copyMetadata(lv.currentTripCount());
        new_loop_body->eraseInput(0);

        // replace the new created loop inputs that won't be updated in the loop with the values from outer blocks
        torch::jit::LoopView new_lv(new_loop);
        if (new_lv.bodyCarriedInputs().size() > lv.bodyCarriedInputs().size()) {
          for (int i = new_lv.bodyCarriedInputs().size() - 1; i >= lv.bodyCarriedInputs().size(); --i) {
            new_loop_body->inputs()[i + 1]->replaceAllUsesWith(old_to_new_g[seg_block.raw_inputs()[i]]);
            new_loop_body->eraseInput(i + 1);
          }
        }

        // adjust the output order
        auto loop_out_idx = 0;
        for (size_t i = 0; i < block_outputs.size(); ++i) {
          if (loop_mapping.count(block_outputs[i])) {
            old_to_new_g[block_outputs[i]] = new_loop_body->outputs()[loop_out_idx++];
          }
        }
        for (int i = new_loop_body->outputs().size() - 1; i >= 0; --i) {
          new_loop_body->eraseOutput(i);
        }

        for (size_t i = 0; i < block_outputs.size(); ++i) {
          new_loop_body->registerOutput(old_to_new_g[block_outputs[i]]);
        }
        for (auto ov : lv.carriedOutputs()) {
          auto no = new_loop->addOutput();
          old_to_new_g[ov] = no;
          no->copyMetadata(ov);
        }

      } else {
        AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
      }
    }
  }

  for (auto& output : block->outputs()) {
    printf("output %s\n", output->debugName().c_str());
    if (old_to_new_g.count(output))
      new_g->registerOutput(old_to_new_g[output]);
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
      //      auto convert_cfg = std::move(cfg.convert_info);
      LOG_INFO(*g << "(LoweringGraph)\n");

      // segment the graph and convert segmented TensorRT block
      //      auto segmented_blocks = partitioning::Partition(g, convert_cfg.input_ranges, cfg.partition_info);
      //      if (segmented_blocks.size() == 1 && segmented_blocks[0].target() == partitioning::SegmentedBlock::kTorch)
      //      {
      //        return mod;
      //      }

      std::unordered_map<torch::jit::Value*, torch::jit::Value*> old_to_new_g;
      int trt_engine_id = 0;
      std::unordered_map<torch::jit::Value*, ir::InputRange> input_ranges;
      for (size_t i = 0; i < g->inputs().size(); ++i) {
        input_ranges.insert({g->inputs()[i], cfg.convert_info.input_ranges[i]});
        //        input_ranges[g->inputs()[i]] = cfg.convert_info.input_ranges[i];
      }
      auto input_ivalues_map = partitioning::generateRandomInputs(input_ranges);
      auto inner_fallback_graph =
          ConstructFallbackBlock(new_mod, g->block(), input_ivalues_map, cfg, trt_engine_id, named_params);
      auto inner_graph = inner_fallback_graph.first;
      auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, new_g, old_to_new_g); };
      new_g->block()->cloneFrom(inner_graph->block(), env);

      //      for (auto& seg_block : segmented_blocks) {
      //        if (seg_block.target() == partitioning::SegmentedBlock::kTensorRT) {
      //          std::vector<conversion::InputRange> input_ranges;
      //          for (auto& shape : seg_block.in_shape()) {
      //            input_ranges.push_back(conversion::InputRange(util::toVec(shape)));
      //          }
      //          // update the input ranges for each segments
      //          convert_cfg.input_ranges = input_ranges;
      //          auto engine = conversion::ConvertBlockToEngine(seg_block.block(), convert_cfg, named_params);
      //          auto temp_g = std::make_shared<torch::jit::Graph>();
      //          AddEngineToGraph(new_mod, temp_g, engine, trt_engine_id++);
      //
      //          seg_block.update_graph(temp_g);
      //          AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
      //        } else {
      //          AddSegmentedBlockToGraph(new_g, seg_block, old_to_new_g);
      //        }
      //      }
      //
      //      for (auto& output : g->outputs()) {
      //        new_g->registerOutput(old_to_new_g[output]);
      //      }

      LOG_INFO(*new_g << "(FallbackGraph)\n");

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
