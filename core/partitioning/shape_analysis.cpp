#include <queue>
#include "ATen/ATen.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/constant_pooling.h"

#include "core/partitioning/partitioning.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

at::Tensor generateSingleInput(
    ir::Input& input,
    c10::optional<at::ScalarType>& type_opt,
    const ir::ShapeMode& shape_mode) {
  nvinfer1::Dims input_shape = input.input_shape;
  if (input.input_is_dynamic) {
    if (shape_mode == ir::ShapeMode::kMIN) {
      input_shape = input.min;
    } else if (shape_mode == ir::ShapeMode::kOPT) {
      input_shape = input.opt;
    } else {
      input_shape = input.max;
    }
  }

  // Initialize min and max ranges for random number selection
  int LoValIncl = 0;
  int HiValExcl = 2;

  auto type = at::kFloat;
  if (type_opt) {
    type = type_opt.value();
  } else {
    LOG_WARNING("Input type for doing shape analysis could not be determined, defaulting to F32");
  }

  // Make the value range for input tensor a uniform (float) distribution
  // over [LoValIncl, HiValExcl), then cast to the desired dtype
  auto in = ((HiValExcl - LoValIncl) * at::rand(util::toVec(input_shape), {at::kCUDA}) + LoValIncl).to(type);

  return in;
}

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, std::vector<ir::Input>>& inputs,
    std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>>& types,
    const ir::ShapeMode& shape_mode) {
  // generate random inputs for running pytorch segments
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> ivalue_map;

  for (auto& input : inputs) {
    if (input.first->type()->kind() == torch::jit::TypeKind::ListType) {
      c10::TypePtr elementType = c10::TensorType::get();
      auto generic_list = c10::impl::GenericList(elementType);
      for (size_t i = 0; i < input.second.size(); i++) {
        auto in = generateSingleInput(input.second[i], types[input.first][i], shape_mode);
        generic_list.push_back(in.clone());
      }
      ivalue_map[input.first] = c10::IValue(generic_list);
    } else if (input.first->type()->kind() == torch::jit::TypeKind::TupleType) {
      // create tuple
      std::vector<torch::jit::IValue> list;
      for (size_t i = 0; i < input.second.size(); i++) {
        auto in = generateSingleInput(input.second[i], types[input.first][i], shape_mode);
        list.push_back(in.clone());
      }
      auto tuple = c10::ivalue::Tuple::create(list); // create tuple ptr
      ivalue_map[input.first] = c10::IValue(tuple);
    } else {
      auto in = generateSingleInput(input.second[0], types[input.first][0], shape_mode);
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
    if ((node->kind().toQualString() == std::string("aten::to")) &&
        ((node->inputs()[1]->node()->output()->type()->kind() == torch::jit::TypeKind::IntType) ||
         (node->inputs()[2]->node()->output()->type()->kind() == torch::jit::TypeKind::IntType))) {
      return node;
    }
    if (node->kind() != torch::jit::prim::Constant && !visited.count(node)) {
      visited.insert(node);
      for (auto input : node->inputs()) {
        q.push(input);
      }
    }
  }
  return nullptr;
}

torch::jit::Node* createCastNode(
    SegmentedBlock& seg_block,
    size_t index,
    bool is_input,
    at::ScalarType dtype,
    std::string device,
    bool force_create_node = false) {
  auto cast_raw_value = is_input ? seg_block.raw_inputs()[index] : seg_block.raw_outputs()[index];
  auto cast_subgraph_value = is_input ? seg_block.inputs()[index] : seg_block.outputs()[index];
  torch::jit::Node* cast_node = getUpstreamCastNode(cast_raw_value);
  auto g = seg_block.g();
  // if we can find upstream aten::to node, we use it's parameters for creating new cast node
  if (cast_node && !force_create_node) {
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;
    value_map.insert({cast_node->inputs()[0], cast_subgraph_value});
    if (!is_input) {
      // if this value is output, we need to cast it to int32
      auto const_val = g->insertConstant(dtype);
      if (cast_node->inputs()[1]->node()->output()->type()->kind() == torch::jit::TypeKind::DeviceObjType) {
        value_map.insert({cast_node->inputs()[2], const_val});
      } else {
        value_map.insert({cast_node->inputs()[1], const_val});
      }
    }
    auto env = [&](torch::jit::Value* v) { return util::getOrAddInputForValue(v, g, value_map); };
    cast_node = g->createClone(cast_node, env);
    //    auto cast_node = g->prependNode(g->createClone(cast_node, env));
  } else {
    // if there is no explicit cast aten::to operation, we need to create a node
    auto const_type = g->insertConstant(dtype);
    auto const_zero = g->insertConstant(0);
    const_zero->setType(torch::jit::BoolType::get());
    auto cuda = g->insertConstant(device);
    cuda->setType(torch::jit::DeviceObjType::get());
    auto none_val = g->insertNode(g->createNone())->output();
    cast_node =
        g->create(torch::jit::aten::to, {cast_subgraph_value, cuda, const_type, const_zero, const_zero, none_val});
  }
  return cast_node;
}

void getSegmentsOutputByRunning(
    SegmentedBlock& seg_block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitioningInfo& partitioning_info,
    const ir::ShapeMode& shape_mode) {
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

  auto target_device = partitioning_info.getGPUDeviceString();

  // auto int64 <=> int32 conversion + int8 <=> int32 conversion for non-quantized models
  if (seg_block.target() == SegmentedBlock::kTorch) {
    // First, check if there is Int64 input
    for (size_t i = 0; i < seg_block.inputs().size(); ++i) {
      if (ivalues_maps[seg_block.raw_inputs()[i]].isTensor()) {
        auto cur_ivalue = ivalues_maps[seg_block.raw_inputs()[i]];
        at::ScalarType t = cur_ivalue.toTensor().scalar_type();
        if (t == at::kLong && partitioning_info.truncate_long_and_double) {
          LOG_DEBUG(
              "Detected graph Long tensor input type during shape analysis, "
              << "inserting aten::to cast to Long to ensure this Torch block receives "
              << "a Long-type tensor input.");
          // we add a cast operation to cast the type to Int64
          auto cast_node = createCastNode(seg_block, i, true, at::kLong, target_device);
          seg_block.g()->prependNode(cast_node);
          seg_block.inputs()[i]->replaceAllUsesAfterNodeWith(cast_node, cast_node->outputs()[0]);
        } else if (t == at::kByte && partitioning_info.cast_int8_inputs) {
          LOG_DEBUG(
              "Detected graph Byte tensor input type during shape analysis, "
              << "inserting aten::to cast to Byte to ensure this Torch block receives "
              << "a Byte-type tensor input.");
          // If the input has type Byte, ensure it is casted to the correct type
          auto cast_node = createCastNode(seg_block, i, true, at::kByte, target_device, /*force_create_node=*/true);
          seg_block.g()->prependNode(cast_node);
          seg_block.inputs()[i]->replaceAllUsesAfterNodeWith(cast_node, cast_node->outputs()[0]);
        }
      }
    }

    for (size_t i = 0; i < seg_block.outputs().size(); ++i) {
      if (ivalues_maps[seg_block.raw_outputs()[i]].isTensor()) {
        auto cur_ivalue = ivalues_maps[seg_block.raw_outputs()[i]];
        at::ScalarType t = cur_ivalue.toTensor().scalar_type();

        // If the output has type Long and truncation was requested, insert truncate
        if (t == at::kLong && partitioning_info.truncate_long_and_double) {
          LOG_DEBUG(
              "Detected graph Long tensor output type during shape analysis, "
              << "inserting aten::to cast to Int to ensure the subsequent TensorRT block "
              << "receives an Int-type tensor input.");
          auto cast_node = createCastNode(seg_block, i, false, at::kInt, target_device);
          seg_block.g()->appendNode(cast_node);
          seg_block.g()->block()->replaceOutput(i, cast_node->outputs()[0]);
        } else if (t == at::kByte && partitioning_info.cast_int8_inputs) {
          LOG_DEBUG(
              "Detected graph Byte tensor output type during shape analysis, "
              << "inserting aten::to cast to Int to ensure the subsequent TensorRT block "
              << "receives an Int-type tensor input.");
          // If the output has type Byte and casting was requested, insert Integer cast
          auto cast_node = createCastNode(seg_block, i, false, at::kInt, target_device, /*force_create_node=*/true);
          seg_block.g()->appendNode(cast_node);
          seg_block.g()->block()->replaceOutput(i, cast_node->outputs()[0]);
        }
      }
    }
  }

  // set input shape for each segmented block so we wil use it in conversion process
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<at::ScalarType> input_types;
  for (size_t i = 0; i < seg_block.inputs().size(); ++i) {
    auto current_input = seg_block.raw_inputs()[i];

    if (ivalues_maps[current_input].isTensor()) {
      // set the input_shape and data_type
      // we can use a temp value here instead of replacing the values in ivalues_map since we only use ivalues_map for
      // shape inference
      auto cur_ivalue = ivalues_maps[current_input];
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
      } else if (dtype && dtype.value() == nvinfer1::DataType::kINT8 && partitioning_info.cast_int8_inputs) {
        // Special case to ensure input IValues to TensorRT engine are not Int8 type if the
        // model itself is not quantized
        cur_ivalue = cur_ivalue.toTensor().to(at::kInt);
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

  seg_block.register_inshapes(input_shapes, shape_mode);
  seg_block.register_intypes(input_types);
}

void runShapeAnalysis(
    PartitioningCtx* ctx,
    torch::jit::Block* block,
    ExampleIValues& example_tensor_map,
    const ir::ShapeMode& shape_mode) {
  // register every segment's input shape, and it's running output IValues
  for (auto& seg_block : ctx->partitioned_blocks[block]) {
    LOG_GRAPH("Running shape analysis on block " << seg_block);
    torch::jit::ConstantPooling(seg_block.g());
    getSegmentsOutputByRunning(seg_block, example_tensor_map, ctx->settings, shape_mode);
  }
  return;
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
