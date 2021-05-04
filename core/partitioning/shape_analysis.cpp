#include "core/partitioning/shape_analysis.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

namespace trtorch {
namespace core {
namespace partitioning {

std::unordered_map<torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<torch::jit::Value*, ir::InputRange>& input_ranges) {
  // generate random inputs for running pytorch segments
  std::unordered_map<torch::jit::Value*, torch::jit::IValue> ivalue_maps;
  std::vector<torch::jit::IValue> random_inputs;
  for (auto& input_range : input_ranges) {
    auto cur_shape = input_range.second.input_shape;
    std::vector<int64_t> shape;
    shape.insert(shape.begin(), std::begin(cur_shape.d), std::begin(cur_shape.d) + cur_shape.nbDims);
    auto in = at::randint(5, shape, {at::kCUDA});
    ivalue_maps[input_range.first] = in.clone();
  }
  return ivalue_maps;
}

void getSegmentsOutputByRunning(
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
  auto schema = util::GenerateGraphSchema(cur_method->name(), copy_g);
  cur_mod.type()->addMethod(cur_method);
  cur_method->setSchema(schema);

  std::vector<torch::jit::IValue> jit_inputs_ivalues;

  // set inputs ivalues, now supports Tensor/Int to pass argumentes between different segments
  for (auto& input : seg_block.raw_inputs()) {
    TRTORCH_CHECK(
        ivalues_maps.count(input),
        "Could not find torch::jit::Value* " << input->debugName() << " in lowering graph for mini graph input.\n");
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
    } else if (input->type()->kind() == torch::jit::TypeKind::TupleType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTuple());
    } else {
      TRTORCH_THROW_ERROR("Unable to find type for value: " << input->debugName() << " to get the ivalues.\n");
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
  std::vector<ir::InputRange> input_shape;
  for (auto& i : seg_block.raw_inputs()) {
    if (ivalues_maps[i].isTensor()) {
      input_shape.push_back(util::toVec(util::toDims(ivalues_maps[i].toTensor().sizes())));
    }
  }

  seg_block.register_inshape(input_shape);
}

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& ivalues_maps) {
  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : segmented_blocks) {
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, ivalues_maps);
  }
  return;
}

} // namespace partitioning
} // namespace core
} // namespace trtorch
