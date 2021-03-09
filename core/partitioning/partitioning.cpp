#include "partitioning.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/module.h"
#include "core/util/prelude.h"


namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Value* getOrAddInputForValue(torch::jit::Value* old_value, std::shared_ptr<torch::jit::Graph> &graph,
                                         std::unordered_map<torch::jit::Value*, torch::jit::Value*> &old_to_new) {
  if (old_to_new.count(old_value) == 0) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = graph->createClone(node, {nullptr});
      graph->block()->prependNode(new_const);
      return new_const->output();
    }
    auto new_value = graph->block()->addInput();
    old_to_new[old_value] = new_value;
    // mapping from new graph input Values to original graph values
    old_to_new[new_value] = old_value;
    new_value->copyMetadata(old_value);
    return new_value;
  } else {
    return old_to_new[old_value];
  }
}

torch::jit::Node* cloneNode(torch::jit::Node* node, std::shared_ptr<torch::jit::Graph> &graph,
                            std::unordered_map<torch::jit::Value*, torch::jit::Value*> &old_to_new) {
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

void registerSegmentInOutShape(SegmentedBlock &seg_block, std::unordered_map<torch::jit::Value*, nvinfer1::Dims> &input_shape_map) {
  // create a module to run the graph
  auto g = seg_block.g();
  auto copy_g = g->copy();
  torch::jit::script::Module cur_mod(c10::QualifiedName("module"));

  auto self = copy_g->insertInput(0, "self_1");
  self->setType(cur_mod.type());

  auto cur_method = cur_mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), copy_g);
  auto schema =  getFunctionSchema(cur_method->name(), copy_g);
  cur_mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;

  // set inputs ivalues
  for (auto &input : seg_block.raw_inputs()) {
    std::vector<int64_t> shape;
    nvinfer1::Dims cur_shape = input_shape_map[input];
    shape.insert(shape.begin(), std::begin(cur_shape.d), std::begin(cur_shape.d) + cur_shape.nbDims);
    auto in = at::randint(5, shape, {at::kCUDA});
    jit_inputs_ivalues.push_back(in.clone());
  }

  std::vector<at::Tensor> jit_results;
  torch::jit::IValue jit_results_ivalues = cur_mod.forward(jit_inputs_ivalues);
  if (jit_results_ivalues.isTensor()) {
    jit_results.push_back(jit_results_ivalues.toTensor());
  } else {
    auto results = jit_results_ivalues.toTuple()->elements();
    for (auto r : results) {
      jit_results.push_back(r.toTensor());
    }
  }

  size_t idx = 0;
  for (auto &output : seg_block.raw_outputs()) {
    input_shape_map[output] = util::toDims(jit_results[idx++].sizes());
  }

  std::vector<nvinfer1::Dims> input_shape;
  for (auto &i : seg_block.raw_inputs()) {
    input_shape.push_back(input_shape_map[i]);
  }

  seg_block.register_inshape(input_shape);
}

std::vector<nvinfer1::Dims> extractNvinfer1Dims(std::vector<conversion::InputRange>& input_ranges) {
  std::vector<nvinfer1::Dims> res;
  for (auto &input_range : input_ranges) {
    res.push_back(input_range.input_shape);
  }
  return res;
}

void registerSegmentsInputsOutputs(std::vector<SegmentedBlock> &segmented_blocks, std::shared_ptr<torch::jit::Graph> g) {
  std::set<torch::jit::Value*> input_values;
  for (auto &seg_block : segmented_blocks) {
    seg_block.registerInputs();
    for (auto &input : seg_block.raw_inputs()) {
      input_values.insert(input);
    }
  }

//  for (auto &graph_input : g->inputs()) {
//    input_values.erase(graph_input);
//  }

  for (auto &graph_output : g->outputs()) {
    input_values.insert(graph_output);
  }

  for (auto &mini_graph_input : input_values) {
    for (auto &seg_block : segmented_blocks) {
      if (std::find(seg_block.raw_inputs().begin(), seg_block.raw_inputs().end(), mini_graph_input)
              == seg_block.raw_inputs().end() && seg_block.contain_raw_input(mini_graph_input)) {
        seg_block.registerOutput(mini_graph_input);
      }
    }
  }

  return;
}

std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g, std::vector<conversion::InputRange>& input_ranges) {
  std::vector<SegmentedBlock> segmented_blocks;

  auto nodes = g->block()->nodes();

  // segment the nodes
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) continue;

    auto block_target = conversion::OpSupported(n) ? SegmentedBlock::kTensorRT : SegmentedBlock::kTorch;

    if (segmented_blocks.empty() || block_target != segmented_blocks.back().target()) {
      SegmentedBlock cur_block(block_target);
      cur_block.appendNode(n);
      segmented_blocks.push_back(cur_block);
    } else {
        segmented_blocks.back().appendNode(n);
    }
  }

  registerSegmentsInputsOutputs(segmented_blocks, g);

  std::vector<nvinfer1::Dims> graph_inputs_shape = extractNvinfer1Dims(input_ranges);
  std::unordered_map<torch::jit::Value*, nvinfer1::Dims> input_shape_map;

  for (size_t i = 0; i < g->inputs().size(); ++i) {
    input_shape_map[g->inputs()[i]] = graph_inputs_shape[i];
  }

  for (auto &seg_block : segmented_blocks) {
    registerSegmentInOutShape(seg_block, input_shape_map);
  }

  return segmented_blocks;
}

}
}
}


