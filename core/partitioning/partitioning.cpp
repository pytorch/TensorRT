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

std::vector<nvinfer1::Dims> registerSegmentInOutShape(SegmentedBlock &seg_block, std::vector<nvinfer1::Dims> &input_shape) {
  auto g = seg_block.g_->copy();
  torch::jit::script::Module cur_mod(c10::QualifiedName("module"));

  auto self = g->insertInput(0, "self_1");
  self->setType(cur_mod.type());

  auto cur_method = cur_mod._ivalue()->compilation_unit()->create_function(c10::QualifiedName("forward"), g);
  auto schema =  getFunctionSchema(cur_method->name(), g);
  cur_mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<int64_t> shape;
  shape.insert(shape.begin(), std::begin(input_shape[0].d), std::begin(input_shape[0].d) + input_shape[0].nbDims);
  auto in = at::randint(5, shape, {at::kCUDA});
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  jit_inputs_ivalues.push_back(in.clone());

  torch::jit::IValue jit_results_ivalues = cur_mod.forward(jit_inputs_ivalues);
  if (!jit_results_ivalues.isTensor()) {
    std::cerr << "Mini graph output is NOT a Tensor!\n";
  }
  auto jit_results_tensor = jit_results_ivalues.toTensor();
  auto output_sizes = jit_results_tensor.sizes();
  for (auto &i : output_sizes) {
    printf("%d\n", i);
  }

  std::vector<nvinfer1::Dims> output_shape;
  output_shape.push_back(util::toDims(output_sizes));
  seg_block.register_inshape(input_shape);
  seg_block.register_outshape(output_shape);

  return output_shape;
}

std::vector<nvinfer1::Dims> extractNvinfer1Dims(std::vector<conversion::InputRange>& input_ranges) {
  std::vector<nvinfer1::Dims> res;
  for (auto &input_range : input_ranges) {
    res.push_back(input_range.input_shape);
  }
  return res;
}

std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g, std::vector<conversion::InputRange>& input_ranges) {
  std::vector<SegmentedBlock> segmented_blocks;

  auto nodes = g->block()->nodes();

  // segment the nodes
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) continue;
    auto block_target = conversion::OpSupported(n) ? SegmentedBlock::kTensorRT : SegmentedBlock::kTorch;

    if (segmented_blocks.empty() || block_target != segmented_blocks.back().target) {
      SegmentedBlock cur_block(block_target);
      cur_block.appendNode(n);
      segmented_blocks.push_back(cur_block);
    } else {
        segmented_blocks.back().appendNode(n);
    }
  }

  std::vector<nvinfer1::Dims> cur_input = extractNvinfer1Dims(input_ranges);
  for (auto &seg_block : segmented_blocks) {
    seg_block.registerOutput();
    cur_input = registerSegmentInOutShape(seg_block, cur_input);
  }

  return segmented_blocks;
}

}
}
}


