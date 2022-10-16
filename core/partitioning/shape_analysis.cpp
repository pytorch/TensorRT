#include <queue>
#include "ATen/ATen.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

#include "core/partitioning/partitioning.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

at::Tensor generateSingleInput(ir::Input& input, c10::optional<at::ScalarType>& type_opt) {
  auto cur_shape = input.input_shape;
  std::vector<int64_t> shape;
  shape.insert(shape.begin(), std::begin(cur_shape.d), std::begin(cur_shape.d) + cur_shape.nbDims);
  // auto type_opt = types[input.first][i];
  auto type = at::kFloat;
  if (type_opt) {
    type = type_opt.value();
  } else {
    LOG_WARNING("Input type for doing shape analysis could not be determined, defaulting to F32");
  }
  auto in = at::randint(5, shape, {at::kCUDA}).to(type);
  // ivalue_map[input.first] = in.clone();
  return in;
}

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, std::vector<ir::Input>>& inputs,
    std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>>& types) {
  // generate random inputs for running pytorch segments
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> ivalue_map;

  for (auto& input : inputs) {
    if (input.first->type()->kind() == torch::jit::TypeKind::ListType) {
      c10::TypePtr elementType = c10::TensorType::get();
      auto generic_list = c10::impl::GenericList(elementType);
      for (size_t i = 0; i < input.second.size(); i++) {
        auto in = generateSingleInput(input.second[i], types[input.first][i]);
        generic_list.push_back(in.clone());
      }
      ivalue_map[input.first] = c10::IValue(generic_list);
    } else if (input.first->type()->kind() == torch::jit::TypeKind::TupleType) {
      // create tuple
      std::vector<torch::jit::IValue> list;
      for (size_t i = 0; i < input.second.size(); i++) {
        auto in = generateSingleInput(input.second[i], types[input.first][i]);
        list.push_back(in.clone());
      }
      auto tuple = c10::ivalue::Tuple::create(list); // create tuple ptr
      ivalue_map[input.first] = c10::IValue(tuple);
    } else {
      auto in = generateSingleInput(input.second[0], types[input.first][0]);
      ivalue_map[input.first] = in.clone();
    }
  }
  return ivalue_map;
}

torch::jit::Node* getUpstreamCastNode(torch::jit::Value* val) {
  std::queue<torch::jit::Value*> q;
  q.push(val);
  std::unordered_set<torch::jit::Node*> visited;
  while (!q.empty()) {
    auto cur_val = q.front();
    q.pop();
    auto node = cur_val->node();
    if (node->kind().toQualString() == std::string("aten::to")) {
      return node;
    }
    if (node->kind() != torch::jit::prim::Constant && !visited.count(node)) {
      visited.insert(node);
      for (auto input : node->inputs()) {
        q.push(input);
      }
    }
  }
}

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitioningInfo& partitioning_info) {
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
    } else if (input->type()->isSubtypeOf(torch::jit::FloatType::get())) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toDouble());
    } else if (input->type()->kind() == torch::jit::TypeKind::ListType) {
      // create list
      jit_inputs_ivalues.push_back(ivalues_maps[input].toList());
      ;
    } else if (input->type()->kind() == torch::jit::TypeKind::TupleType) {
      // create tuple
      jit_inputs_ivalues.push_back(ivalues_maps[input].toTuple());
    } else if (input->type()->kind() == torch::jit::TypeKind::NumberType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toScalar());
    } else if (input->type()->kind() == torch::jit::TypeKind::DictType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toGenericDict());
    } else if (input->type()->kind() == torch::jit::TypeKind::DeviceObjType) {
      jit_inputs_ivalues.push_back(ivalues_maps[input].toDevice());
    } else {
      TORCHTRT_THROW_ERROR(
          "Expected to find type " << input->type()->str() << " for value " << input->debugName()
                                   << " but get nothing. ");
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

  // auto int64 <=> int32 conversion
  if (seg_block.target() == SegmentedBlock::kTorch) {
    // Firstly, check if there is Int64 input
    for (size_t i = 0; i < seg_block.inputs().size(); ++i) {
      if (ivalues_maps[seg_block.raw_inputs()[i]].isTensor()) {
        auto cur_ivalue = ivalues_maps[seg_block.raw_inputs()[i]];
        at::ScalarType t = cur_ivalue.toTensor().scalar_type();
        if (t == at::kLong) {
          // we add a cast operation to cast the type to Int64
          auto inner_g = seg_block.g();
          torch::jit::Node* cast_node = getUpstreamCastNode(seg_block.raw_inputs()[i]);
          std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;
          value_map.insert({cast_node->inputs()[0], seg_block.inputs()[i]});
          auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, inner_g, value_map); };
          auto new_cast_node = inner_g->prependNode(inner_g->createClone(cast_node, env));
          seg_block.inputs()[i]->replaceAllUsesAfterNodeWith(new_cast_node, new_cast_node->outputs()[0]);
        }
      }
    }
    // TODO: This part might be necessary for some model, now checkint to verify
    //    for (size_t i = 0; i < seg_block.outputs().size(); ++i) {
    //      if (ivalues_maps[seg_block.raw_outputs()[i]].isTensor()) {
    //        auto cur_ivalue = ivalues_maps[seg_block.raw_outputs()[i]];
    //        at::ScalarType t = cur_ivalue.toTensor().scalar_type();
    //        if (t == at::kLong) {
    //          auto inner_g = seg_block.g();
    //          torch::jit::Node* cast_node = getUpstreamCastNode(seg_block.raw_outputs()[i]);
    //          std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;
    //          value_map.insert({cast_node->inputs()[0], seg_block.outputs()[i]});
    //          auto const_val = inner_g->insertConstant(3);
    //          value_map.insert({cast_node->inputs()[1], const_val});
    //          auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, inner_g, value_map); };
    //          auto new_cast_node = inner_g->appendNode(inner_g->createClone(cast_node, env));
    //
    //        }
    //      }
    //    }
  }

  // set input shape for each segmented block so we wil use it in conversion process
  std::vector<ir::Input> input_shapes;
  std::vector<at::ScalarType> input_types;
  for (size_t i = 0; i < seg_block.inputs().size(); ++i) {
    if (ivalues_maps[seg_block.raw_inputs()[i]].isTensor()) {
      // set the input_shape and data_type
      // we can use a temp value here instead of replacing the values in ivalues_map since we only use ivalues_map for
      // shape inference
      auto cur_ivalue = ivalues_maps[seg_block.raw_inputs()[i]];
      at::ScalarType t = cur_ivalue.toTensor().scalar_type();

      if (!partitioning_info.truncate_long_and_double && (t == at::kLong || t == at::kDouble)) {
        TORCHTRT_THROW_ERROR(
            "Unable to process subgraph input type of at::kLong/at::kDouble, try to compile model with truncate_long_and_double enabled");
      } else if (partitioning_info.truncate_long_and_double && t == at::kLong) {
        cur_ivalue = cur_ivalue.toTensor().to(at::kInt);
        LOG_WARNING("Truncating graph input type from at::kLong to at::kInt");
      } else if (partitioning_info.truncate_long_and_double && t == at::kDouble) {
        cur_ivalue = cur_ivalue.toTensor().to(at::kFloat);
        LOG_WARNING("Truncating graph input type from at::kDouble to at::kFloat");
      }
      c10::optional<nvinfer1::DataType> dtype = util::optTypeMetaToTRTDataType(cur_ivalue.toTensor().dtype());
      if (dtype == c10::nullopt) {
        TORCHTRT_THROW_ERROR("Unsupported input data type " << cur_ivalue.toTensor().dtype());
      }
      if (cur_ivalue.toTensor().sizes().size() == 0) {
        // handle Scalar types, which has sizes of []
        input_shapes.push_back(util::toVec(util::toDims(c10::List<int64_t>({1}))));
      } else {
        input_shapes.push_back(util::toVec(util::toDims(cur_ivalue.toTensor().sizes())));
      }
      input_types.push_back(cur_ivalue.toTensor().scalar_type());
    }
    // TODO: tuple and list inputs in subgraph
  }

  seg_block.register_inshapes(input_shapes);
  seg_block.register_intypes(input_types);
}

void runShapeAnalysis(PartitioningCtx* ctx, torch::jit::Block* block, ExampleIValues& example_tensor_map) {
  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : ctx->partitioned_blocks[block]) {
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, example_tensor_map, ctx->settings);
  }
  return;
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
