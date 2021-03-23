#include "partitioning.h"
#include "core/conversion/evaluators/eval_util.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/module.h"

namespace trtorch {
namespace core {
namespace partitioning {

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
    if (!ivalues_maps.count(input)) {
      std::cerr << "could find graph input ivalues\n";
    }
    if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTensor());
    } else if (input->type()->isSubtypeOf(torch::jit::IntType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toInt());
    } else if (input->type()->isSubtypeOf(torch::jit::BoolType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toBool());
    } else if (input->type()->isSubtypeOf(torch::jit::ListType::ofTensors())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toList());
    } else {
      std::cerr << "Currently not support the type cast for input type " << input->type()->str() << ".\n";
    }
  }

  // run segments to get outputs for later segments input shape, and other arguments such as Int
  std::vector<torch::jit::IValue> jit_results;
  torch::jit::IValue jit_results_ivalues = cur_mod.forward(jit_inputs_ivalues);
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

void registerSegmentsInputsOutputs(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::shared_ptr<torch::jit::Graph> g) {
  // find the corresponding raw values in original global graph for this segmented block's inputs/outputs
  std::set<torch::jit::Value*> input_values;
  for (auto& seg_block : segmented_blocks) {
    seg_block.registerInputs();
    for (auto& input : seg_block.raw_inputs()) {
      input_values.insert(input);
    }
  }

  for (auto& graph_output : g->outputs()) {
    input_values.insert(graph_output);
  }

  for (auto& mini_graph_input : input_values) {
    for (auto& seg_block : segmented_blocks) {
      if (std::find(seg_block.raw_inputs().begin(), seg_block.raw_inputs().end(), mini_graph_input) ==
              seg_block.raw_inputs().end() &&
          seg_block.contain_raw_input(mini_graph_input)) {
        seg_block.registerOutput(mini_graph_input);
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

std::vector<SegmentedBlock> segment_graph(
    std::shared_ptr<torch::jit::Graph> g,
    std::vector<conversion::InputRange>& input_ranges,
    const conversion::TorchFallback& fallback_info) {
  auto min_block_size = fallback_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_operators(
      fallback_info.forced_fallback_operators.begin(), fallback_info.forced_fallback_operators.end());
  std::vector<SegmentedBlock> segmented_blocks;

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

  // register input/output torch::jit::Value for segmetned graphs
  registerSegmentsInputsOutputs(segmented_blocks, g);

  // store the mapping from lowering graph torch::jit::Value => torch::jit::IValue that we get by running segments
  std::unordered_map<torch::jit::Value*, torch::jit::IValue> ivalues_maps;

  std::vector<torch::jit::IValue> random_inputs = generateRandomInputs(input_ranges);
  for (size_t i = 0; i < g->inputs().size(); ++i) {
    ivalues_maps[g->inputs()[i]] = random_inputs[i];
  }

  // register every segment's input shape, and it's running output Ivalues
  for (auto& seg_block : segmented_blocks) {
    registerSegmentInOutIValues(seg_block, ivalues_maps);
  }

  return segmented_blocks;
}

} // namespace partitioning
} // namespace core
} // namespace trtorch
