#include "partitioning.h"
#include <queue>
#include "core/conversion/evaluators/eval_util.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

namespace trtorch {
namespace core {
namespace partitioning {

inline bool isTensorOrTensorList(torch::jit::Value* val) {
  return val->type()->isSubtypeOf(torch::jit::TensorType::get()) ||
      val->type()->isSubtypeOf(torch::jit::ListType::ofTensors());
}

struct usage_info {
  int produce_id = -1;
  std::vector<int> torch_use_id;
  std::vector<int> tensorrt_use_id;
};

torch::jit::Value* getOrAddInputForValue(
    torch::jit::Value* old_value,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  if (old_to_new.count(old_value) == 0) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = graph->createClone(node, {nullptr});
      graph->block()->prependNode(new_const);
      return new_const->output();
    }
    auto new_value = graph->block()->addInput();
    old_to_new[old_value] = new_value;
    new_value->copyMetadata(old_value);
    // mapping from new graph input Values to original graph values
    old_to_new[new_value] = old_value;
    return new_value;
  } else {
    return old_to_new[old_value];
  }
}

torch::jit::Node* cloneNode(
    torch::jit::Node* node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  auto* block = graph->block();
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v, graph, old_to_new); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(graph->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new[oo] = no;
  }
  return new_node;
}

c10::FunctionSchema getFunctionSchema(std::string method_name, std::shared_ptr<torch::jit::Graph>& g) {
  std::vector<c10::Argument> args;
  for (auto in : g->inputs()) {
    args.push_back(c10::Argument(in->debugName(), in->type()));
  }

  std::vector<c10::Argument> returns;
  for (auto out : g->outputs()) {
    returns.push_back(c10::Argument(out->debugName(), out->type()));
  }

  return c10::FunctionSchema(method_name, method_name, args, returns);
}

void registerSegmentInOutIValues(
    SegmentedBlock& seg_block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps) {
  // create a module to run the graph
  auto g = seg_block.g();
  auto copy_g = g->copy();

  // create tuple for multiple outputs
  if (seg_block.raw_outputs().size() > 1) {
    auto new_output_node = copy_g->appendNode(copy_g->createTuple(copy_g->outputs()));
    for (int idx = copy_g->outputs().size() - 1; idx >= 0; --idx) {
      copy_g->eraseOutput(idx);
    }
    copy_g->registerOutput(new_output_node->outputs()[0]);
  }

  torch::jit::script::Module cur_mod(c10::QualifiedName("module"));

  auto self = copy_g->insertInput(0, "self_1");
  self->setType(cur_mod.type());

  auto cur_method = cur_mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), copy_g);
  auto schema = getFunctionSchema(cur_method->name(), copy_g);
  cur_mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;

  // set inputs ivalues, now supports Tensor/Int to pass argumentes between different segments
  for (auto& input : seg_block.raw_inputs()) {
    TRTORCH_CHECK(ivalues_maps.count(input), "Could not find mini graph input IValue " << input->debugName());
    if (input->node()->kind() == torch::jit::prim::Param) {
      jit_inputs_ivalues.push_back(ivalues_maps[input]);
    } else if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTensor());
    } else if (input->type()->isSubtypeOf(torch::jit::IntType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toInt());
    } else if (input->type()->isSubtypeOf(torch::jit::BoolType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toBool());
    } else if (input->type()->kind() == torch::jit::TypeKind::ListType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toList());
    } else {
      TRTORCH_CHECK(input->type()->kind() == torch::jit::TypeKind::TupleType, "Input for mini graph is not TupleType.");
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTuple());
    }
  }

  // run segments to get outputs for later segments input shape, and other arguments such as Int
  std::vector<torch::jit::IValue> jit_results;
  printf("before forward\n");
  torch::jit::IValue jit_results_ivalues = cur_mod.forward(jit_inputs_ivalues);
  printf("after forward\n");

  if (jit_results_ivalues.isTuple()) {
    auto results = jit_results_ivalues.toTuple()->elements();
    for (auto r : results) {
      jit_results.push_back(r);
    }
  } else {
    jit_results.push_back(jit_results_ivalues);
  }

  size_t idx = 0;
  for (auto& output : seg_block.raw_outputs()) {
    ivalues_maps[output] = jit_results[idx++];
  }

  // set input shape for each segmented block so we wil use it in conversion process
  std::vector<nvinfer1::Dims> input_shape;
  for (auto& i : seg_block.raw_inputs()) {
    if (ivalues_maps[i].isTensor()) {
      input_shape.push_back(util::toDims(ivalues_maps[i].toTensor().sizes()));
    }
  }

  seg_block.register_inshape(input_shape);
}

std::vector<torch::jit::IValue> generateRandomInputs(std::vector<conversion::InputRange>& input_ranges) {
  // generate random inputs for running pytorch segments
  std::vector<torch::jit::IValue> random_inputs;
  for (auto& input_range : input_ranges) {
    auto cur_shape = input_range.input_shape;
    std::vector<int64_t> shape;
    shape.insert(shape.begin(), std::begin(cur_shape.d), std::begin(cur_shape.d) + cur_shape.nbDims);
    auto in = at::randint(5, shape, {at::kCUDA});
    random_inputs.push_back(in.clone());
  }
  return random_inputs;
}

void registerSegmentsOutputs(std::vector<SegmentedBlock>& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  std::set<torch::jit::Value*> input_values;
  for (auto& seg_block : segmented_blocks) {
    for (auto& input : seg_block.raw_inputs()) {
      input_values.insert(input);
    }
  }

  for (auto& graph_output : g->outputs()) {
    input_values.insert(graph_output);
  }

  // should be careful here because some in-place operations don't return any values, there is no output for this kind
  // of segment identify the output for each mini-graph by checking if any value in this graph is used later we
  // shouldn't register nonTensor output for TensorRT segments
  for (auto& seg_block : segmented_blocks) {
    for (auto& mini_graph_input : input_values) {
      if (std::find(seg_block.raw_inputs().begin(), seg_block.raw_inputs().end(), mini_graph_input) ==
              seg_block.raw_inputs().end() &&
          seg_block.contain_raw_value(mini_graph_input)) {
        if (!isTensorOrTensorList(mini_graph_input) && seg_block.target() == SegmentedBlock::kTensorRT)
          continue;
        seg_block.registerOutput(mini_graph_input);
      }
    }
    // if no output, then register the last node's output as current graph's output
    if (seg_block.raw_outputs().empty()) {
      // for Torch segments, register input as output
      if (seg_block.target() == SegmentedBlock::kTorch) {
        seg_block.registerOutput(seg_block.raw_inputs()[0]);
      } else {
        // for TensorRT segments, register last nonInput Tensor outputs
        for (int i = seg_block.raw_nodes().size() - 1; i >= 0; --i) {
          for (auto node_output : seg_block.raw_nodes()[i]->outputs()) {
            if (isTensorOrTensorList(node_output))
              seg_block.registerOutput(node_output);
          }
          if (!seg_block.raw_outputs().empty())
            break;
        }
      }
    }
  }
  // erase segments which still have no output
  segmented_blocks.erase(
      std::remove_if(
          segmented_blocks.begin(),
          segmented_blocks.end(),
          [](SegmentedBlock& seg_block) { return seg_block.raw_outputs().empty(); }),
      segmented_blocks.end());

  return;
}

std::vector<torch::jit::Node*> getDependencyNodes(std::vector<torch::jit::Value*>& vals) {
  // using bfs to get the DAG dependency nodes for input value
  std::queue<torch::jit::Value*, std::deque<torch::jit::Value*>> q(
      std::deque<torch::jit::Value*>(vals.begin(), vals.end()));
  std::unordered_set<torch::jit::Node*> visited;
  std::vector<torch::jit::Node*> stk;
  while (!q.empty()) {
    auto cur_val = q.front();
    q.pop();
    auto node = cur_val->node();
    if (node->kind() != torch::jit::prim::Constant && !visited.count(node)) {
      stk.push_back(node);
      for (auto input : node->inputs()) {
        if (!isTensorOrTensorList(input)) {
          q.push(input);
        }
      }
    }
  }
  std::reverse(stk.begin(), stk.end());
  return stk;
}

SegmentedBlock injectNodesForNonTensorInputs(SegmentedBlock& seg_block) {
  // reconstruct segmented_block if this block requires nonTensor input
  std::vector<torch::jit::Value*> nontensor_inputs;
  for (auto input : seg_block.raw_inputs()) {
    if (!isTensorOrTensorList(input)) {
      nontensor_inputs.push_back(input);
    }
  }
  std::vector<torch::jit::Node*> new_block_nodes = getDependencyNodes(nontensor_inputs);
  new_block_nodes.insert(new_block_nodes.end(), seg_block.raw_nodes().begin(), seg_block.raw_nodes().end());
  return SegmentedBlock(seg_block.target(), new_block_nodes);
}

void resolveNonTensorInputs(std::vector<SegmentedBlock>& segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
  // for NonTensor inputs in TensorRT segments, count the usages on Torch segments and TensorRT segments
  std::unordered_map<torch::jit::Value*, usage_info> usage_counts;
  for (int i = segmented_blocks.size() - 1; i >= 0; --i) {
    for (auto input : segmented_blocks[i].raw_inputs()) {
      if (!isTensorOrTensorList(input)) {
        segmented_blocks[i].target() == SegmentedBlock::kTorch ? usage_counts[input].torch_use_id.push_back(i)
                                                               : usage_counts[input].tensorrt_use_id.push_back(i);
      }
    }
    for (auto& use : usage_counts) {
      if (segmented_blocks[i].contain_raw_value(use.first)) {
        use.second.produce_id = i;
      }
    }
  }
  std::unordered_set<int> updated_segments;
  for (auto& use : usage_counts) {
    auto use_info = use.second;
    // if the segment that produce this nonTensor value is kTensorRT but consumed in kTorch, inject nodes in the first
    // kTorch segments
    if (segmented_blocks[use_info.produce_id].target() == SegmentedBlock::kTensorRT && !use_info.torch_use_id.empty()) {
      int first_torch_id = use_info.torch_use_id.front();
      if (!updated_segments.count(first_torch_id)) {
        auto new_torch_block = injectNodesForNonTensorInputs(segmented_blocks[first_torch_id]);
        segmented_blocks[first_torch_id] = new_torch_block;
        updated_segments.insert(first_torch_id);
      }
    } else {
      // KTensorRT segments always need to inject nodes for the nonTensor inputs
      for (int i : use_info.tensorrt_use_id) {
        if (!updated_segments.count(i)) {
          auto new_seg_block = injectNodesForNonTensorInputs(segmented_blocks[i]);
          segmented_blocks[i] = new_seg_block;
          updated_segments.insert(i);
        }
      }
    }
  }
  return;
}

void construct_segments(
    std::vector<torch::jit::Node*>& pytorch_nodes,
    std::vector<torch::jit::Node*>& tensorrt_nodes,
    std::vector<SegmentedBlock>& segmented_blocks,
    size_t min_block_size) {
  // construct segmented blocks according to min_block_size and consecutive nodes
  if (!tensorrt_nodes.empty()) {
    if (tensorrt_nodes.size() < min_block_size) {
      pytorch_nodes.insert(pytorch_nodes.end(), tensorrt_nodes.begin(), tensorrt_nodes.end());
    } else {
      if (!pytorch_nodes.empty())
        segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
      segmented_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
      pytorch_nodes.clear();
    }
    tensorrt_nodes.clear();
  }
}

void segment_graph(
    std::shared_ptr<torch::jit::Graph> g,
    const conversion::TorchFallback& fallback_info,
    std::vector<SegmentedBlock>& segmented_blocks) {
  auto min_block_size = fallback_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_operators(
      fallback_info.forced_fallback_operators.begin(), fallback_info.forced_fallback_operators.end());

  auto nodes = g->block()->nodes();

  // segment the nodes
  std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant)
      continue;

    std::string node_string(n->kind().toQualString());
    if (conversion::OpSupported(n) && !forced_fallback_operators.count(node_string)) {
      tensorrt_nodes.push_back(n);
    } else {
      construct_segments(pytorch_nodes, tensorrt_nodes, segmented_blocks, min_block_size);
      pytorch_nodes.push_back(n);
    }
  }
  construct_segments(pytorch_nodes, tensorrt_nodes, segmented_blocks, min_block_size);
  if (!pytorch_nodes.empty()) {
    segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
  }
}

std::vector<SegmentedBlock> Partition(
    std::shared_ptr<torch::jit::Graph> g,
    std::vector<conversion::InputRange>& input_ranges,
    const conversion::TorchFallback& fallback_info) {
  // segment lowering global graph into blocks
  std::vector<SegmentedBlock> segmented_blocks;
  segment_graph(g, fallback_info, segmented_blocks);

  // resolve nonTensor inputs/outputs
  resolveNonTensorInputs(segmented_blocks, g);

  // register input/output torch::jit::Value for segmented graphs
  registerSegmentsOutputs(segmented_blocks, g);

  // store the mapping from lowering graph torch::jit::Value => torch::jit::IValue that we get by running segments
  std::unordered_map<torch::jit::Value*, torch::jit::IValue> ivalues_maps;
  std::vector<torch::jit::IValue> random_inputs = generateRandomInputs(input_ranges);
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    ivalues_maps[g->inputs()[i]] = random_inputs[i];
  }

  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : segmented_blocks) {
    torch::jit::ConstantPooling(seg_block.g());
    registerSegmentInOutIValues(seg_block, ivalues_maps);
  }

  return segmented_blocks;
}

} // namespace partitioning
} // namespace core
} // namespace trtorch
