#include "core/partitioning/shape_analysis.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, ir::Input>& inputs,
    std::unordered_map<const torch::jit::Value*, c10::optional<at::ScalarType>>& types) {
  // generate random inputs for running pytorch segments
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> ivalue_map;

  uint64_t in_i = 0;
  for (auto& input : inputs) {
    auto cur_shape = input.second.input_shape;
    std::vector<int64_t> shape;
    shape.insert(shape.begin(), std::begin(cur_shape.d), std::begin(cur_shape.d) + cur_shape.nbDims);
    auto type_opt = types[input.first];
    auto type = at::kFloat;
    if (type_opt) {
      type = type_opt.value();
    } else {
      LOG_WARNING("Input type for doing shape analysis could not be determined, defaulting to F32");
    }
    auto in = at::randint(5, shape, {at::kCUDA}).to(type);
    ivalue_map[input.first] = in.clone();
    in_i++;
  }
  return ivalue_map;
}

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitionInfo& partition_info) {
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
    TORCHTRT_CHECK(
        ivalues_maps.count(input),
        "Could not find torch::jit::Value* " << input->debugName() << " produced from "
                                             << util::node_info(input->node())
                                             << " in lowering graph for mini graph input.\n");
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
      TORCHTRT_THROW_ERROR("Unable to find type for value: " << input->debugName() << " to get the ivalues.\n");
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
  std::vector<ir::Input> input_shapes;
  std::vector<at::ScalarType> input_types;
  for (auto& i : seg_block.raw_inputs()) {
    if (ivalues_maps[i].isTensor()) {
      // set the input_shape and data_type
      at::ScalarType t = ivalues_maps[i].toTensor().scalar_type();
      if (!partition_info.truncate_long_and_double && (t == at::kLong || t == at::kDouble)) {
        TORCHTRT_THROW_ERROR(
            "Unable to process subgraph input type of at::kLong/at::kDouble, try to compile model with truncate_long_and_double enabled");
      } else if (partition_info.truncate_long_and_double && t == at::kLong) {
        ivalues_maps[i] = ivalues_maps[i].toTensor().to(at::kInt);
        LOG_WARNING("Truncating graph input type from at::kLong to at::kInt");
      } else if (partition_info.truncate_long_and_double && t == at::kDouble) {
        ivalues_maps[i] = ivalues_maps[i].toTensor().to(at::kFloat);
        LOG_WARNING("Truncating graph input type from at::kDouble to at::kFloat");
      }
      c10::optional<nvinfer1::DataType> dtype = util::optTypeMetaToTRTDataType(ivalues_maps[i].toTensor().dtype());
      if (dtype == c10::nullopt) {
        TORCHTRT_THROW_ERROR("Unsupported input data type " << ivalues_maps[i].toTensor().dtype());
      }
      if (ivalues_maps[i].toTensor().sizes().size() == 0) {
        // handle Scalar types, which has sizes of []
        input_shapes.push_back(util::toVec(util::toDims(c10::List<long int>({1}))));
      } else {
        input_shapes.push_back(util::toVec(util::toDims(ivalues_maps[i].toTensor().sizes())));
      }
      input_types.push_back(ivalues_maps[i].toTensor().scalar_type());
    }
  }

  seg_block.register_inshapes(input_shapes);
  seg_block.register_intypes(input_types);
}

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& example_tensor_map,
    const PartitionInfo& partition_info) {
  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : segmented_blocks) {
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, example_tensor_map, partition_info);
  }
  return;
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
