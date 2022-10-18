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

namespace torch_tensorrt {
namespace core {

void AddEngineToGraph(
    torch::jit::script::Module mod,
    std::shared_ptr<torch::jit::Graph>& g,
    const std::string& serialized_engine,
    runtime::CudaDevice& device_info,
    std::string engine_id = "",
    bool fallback = false) {
  auto engine_ptr = c10::make_intrusive<runtime::TRTEngine>(
      mod._ivalue()->name() + "_engine_" + engine_id, serialized_engine, device_info);
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
  // Go through Lowering to simplify graph
  auto graph_and_parameters = lowering::Lower(mod, method_name, lowering::LowerInfo());

  auto g = graph_and_parameters.first;
  LOG_DEBUG(*g << "(CheckMethodOperatorSupport)\n");

  return conversion::VerifyConverterSupportForBlock(g->block());
}

partitioning::GraphAndMapping BuildHybridGraph(
    torch::jit::script::Module& new_mod,
    torch::jit::Block* block,
    CompileSpec cfg,
    ir::StaticParams static_params,
    ir::CollectionTypeMap first_use_types) {
  auto convert_info = cfg.convert_info;
  auto partitioning_info = cfg.partitioning_info;

  auto partitioning_ctx = partitioning::PartitioningCtx(block, partitioning_info);
  // auto collection_input_ivalues_map =
      // partitioning::generateRandomInputs(partitioning_info.collection_input_spec_map, first_use_types);
  partitioning_ctx.input_types_map = first_use_types;
  partitioning::partition(&partitioning_ctx);

  for (auto& partitioned_block : partitioning_ctx.partitioned_blocks) {
    partitioning::PartitionedGraph& segmented_blocks = partitioned_block.second;

    for (auto& seg_block : segmented_blocks) {
      LOG_INFO("Block segment:" << seg_block);
      std::ostringstream trt_engine_id;
      trt_engine_id << reinterpret_cast<const int*>(&seg_block);

      if (seg_block.target() == partitioning::SegmentedBlock::kTensorRT) {
        // auto shapes = seg_block.in_shapes();
        // auto types = seg_block.in_types();
        // std::vector<ir::Input> inputs;
        // for (size_t i = 0; i < shapes.size(); i++) {
        //   auto in = ir::Input(shapes[i]);
        //   in.dtype = util::ScalarTypeToTRTDataType(types[i]);
        //   inputs.push_back(in);
        // }
        auto inputs = seg_block.construct_inputs_spec();
        LOG_DEBUG("============ INPUTS: " << inputs);
        // update the input ranges for each segments
        convert_info.inputs = ir::associate_specs_with_inputs(seg_block.g(), inputs, static_params);

        // TODO mapping Inputs Ivalue to flatten one here
        auto engine = conversion::ConvertBlockToEngine(seg_block.block(), convert_info, static_params);
        auto temp_g = std::make_shared<torch::jit::Graph>();
        auto device_spec = convert_info.engine_settings.device;
        auto cuda_device = runtime::CudaDevice(device_spec.gpu_id, device_spec.device_type);
        AddEngineToGraph(new_mod, temp_g, engine, cuda_device, trt_engine_id.str(), true);

        seg_block.update_graph(temp_g);
      }
    }
  }

  return partitioning::stitch(&partitioning_ctx, block);
}

void MapInputsAndDetermineDTypes(
    CompileSpec& cfg,
    std::shared_ptr<torch::jit::Graph>& g,
    ir::StaticParams& static_params,
    ir::CollectionTypeMap& first_use_type_map) {
  cfg.convert_info.collection_input_spec_map =
      std::move(ir::associate_specs_with_collection_inputs(g, cfg.graph_inputs, static_params));
  cfg.partitioning_info.collection_input_spec_map =
      ir::CollectionInputSpecMap(cfg.convert_info.collection_input_spec_map);

  auto collection_inputs = ir::get_collection_inputs(g, static_params);
  LOG_DEBUG(
      "In MapInputsAndDetermineDTypes, the g->inputs() size is "
      << g->inputs().size() << ", CollectionInputSpecMap size is" << collection_inputs.size());

  for (auto in : collection_inputs) {
    std::vector<ir::Input>& spec = cfg.convert_info.collection_input_spec_map.find(in)->second;
    std::vector<c10::optional<at::ScalarType>> est_type_opt;

    auto est_it = first_use_type_map.find(in);
    if (est_it != first_use_type_map.end()) {
      est_type_opt = first_use_type_map.find(in)->second;
    }
    // traverse elements in est_type_out and spec
    for (size_t i = 0; i < est_type_opt.size(); i++) {
      if (est_type_opt[i] && !spec[i].dtype_is_user_defined) {
        // If we can calculate the type from the graph and the type was not defined by the user then use the calculated
        // type
        LOG_INFO(
            "Since input type is not explicitly defined, infering using first tensor calculation\n  Inferred input "
            << in->debugName() << " has type " << est_type_opt[i].value());
        spec[i].dtype = util::ScalarTypeToTRTDataType(est_type_opt[i].value());
      } else if (!est_type_opt[i] && !spec[i].dtype_is_user_defined) {
        // If we cannot calculate the type and the user did not define the type, then default to FP32
        LOG_WARNING(
            "Cannot infer input type from calcuations in graph for input "
            << in->debugName() << ". Assuming it is Float32. If not, specify input type explicity");
        spec[i].dtype = nvinfer1::DataType::kFLOAT;
      } else if (spec[i].dtype_is_user_defined && cfg.partitioning_info.enabled) {
        if (!est_type_opt[i]) {
          LOG_INFO("Cannot infer input tensor dtype in graph, compiler is going to use the user setting");
          std::stringstream ss;
          ss << "For input " << in->debugName() << ", found user specified input dtype as ";
          ss << cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype;
          ss << ". The compiler is going to use the user setting "
             << cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype;
          auto warn_str = ss.str();
          LOG_WARNING(warn_str);
          // Overwrite type map with user settings
          first_use_type_map[in][i] = {
              util::TRTDataTypeToScalarType(cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype)};

        } else {
          if (util::TRTDataTypeToScalarType(cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype) !=
              est_type_opt[i].value()) {
            std::stringstream ss;
            ss << "For input " << in->debugName() << ", found user specified input dtype as ";
            ss << cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype;
            ss << ", however when inspecting the graph, the input type expected was inferred to be ";
            ss << est_type_opt[i].value() << std::endl;
            ss << "The compiler is going to use the user setting "
               << cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype;
            ss << "\nThis conflict may cause an error at runtime due to partial compilation being enabled and therefore\n";
            ss << "compatibility with PyTorch's data type convention is required.\n";
            ss << "If you do indeed see errors at runtime either:\n";
            ss << "- Remove the dtype spec for " << in->debugName() << std::endl;
            ss << "- Disable partial compilation by setting require_full_compilation to True";
            auto warn_str = ss.str();
            LOG_WARNING(warn_str);
            // Overwrite type map with user settings
            first_use_type_map[in][i] = {
                util::TRTDataTypeToScalarType(cfg.convert_info.collection_input_spec_map.find(in)->second[i].dtype)};
          }
        }
      } else {
        // The user defined the type so no changes are necessary
      }
    }
  }
  // }
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod, std::string method_name, CompileSpec cfg) {
  // Go through Lowering to simplify graph and extract weight parameters
  auto graph_and_parameters = lowering::Lower(mod, method_name, cfg.lower_info);

  auto g = graph_and_parameters.first;
  TORCHTRT_CHECK(
      conversion::VerifyConverterSupportForBlock(g->block()),
      "Not all operations in graph are supported by the compiler");
  auto params = graph_and_parameters.second;
  auto static_params = ir::get_static_params(g->inputs(), params);
  // Infer the type of an input from the weights of the calculation
  auto first_use_types = ir::get_block_first_calc_dtypes_opt_collection(g->block());

  MapInputsAndDetermineDTypes(cfg, g, static_params, first_use_types);

  auto engine = conversion::ConvertBlockToEngine(g->block(), cfg.convert_info, static_params);

  return engine;
}

torch::jit::Module CompileGraph(const torch::jit::Module& mod, CompileSpec cfg) {
  torch::jit::Module new_mod(mod._ivalue()->name() + "_trt");

  auto device_spec = cfg.convert_info.engine_settings.device;
  auto cuda_device = runtime::CudaDevice(device_spec.gpu_id, device_spec.device_type);

  for (const torch::jit::Method& method : mod.get_methods()) {
    if (method.name().compare("forward") == 0) {
      auto new_g = std::make_shared<torch::jit::Graph>();

      auto graph_and_parameters = lowering::Lower(mod, method.name(), cfg.lower_info);

      auto g = graph_and_parameters.first;
      auto params = graph_and_parameters.second;
      auto static_params = ir::get_static_params(g->inputs(), params);
      // Infer the type of an input from the weights of the calculation
      auto first_use_types = ir::get_block_first_calc_dtypes_opt_collection(g->block());

      MapInputsAndDetermineDTypes(cfg, g, static_params, first_use_types);
      auto isBlockConvertible = conversion::VerifyConverterSupportForBlock(g->block(), true);
      auto outputIsCollection = conversion::OutputIsCollection(g->block());
      if (cfg.partitioning_info.enabled &&
          (cfg.lower_info.forced_fallback_modules.size() == 0 &&
           cfg.partitioning_info.forced_fallback_operators.size() == 0 && isBlockConvertible) &&
          !outputIsCollection) {
        LOG_INFO("Skipping partitioning since model is fully supported");
      }

      if (cfg.partitioning_info.enabled &&
          (!(cfg.lower_info.forced_fallback_modules.size() == 0 &&
             cfg.partitioning_info.forced_fallback_operators.size() == 0 && isBlockConvertible) ||
           outputIsCollection)) {
        auto graph_and_mapping = BuildHybridGraph(new_mod, g->block(), cfg, static_params, first_use_types);
        new_g = graph_and_mapping.first;
        // renaming the input name of graph after fallback to ensure pytorch deserialize it correctly
        for (size_t i = 0; i < new_g->inputs().size(); ++i) {
          new_g->inputs()[i]->setDebugName(std::string("input_") + std::to_string(i));
        }
        LOG_INFO(*new_g << "(GraphAfterFallback)");

        // if there is no tensorrt engine self in fallback graph, there is no conversion, we just return the initial
        // module
        if (new_g->inputs()[0]->type()->str().find("__torch__") == std::string::npos) {
          LOG_WARNING("Didn't generate any TensorRT engines, the compiler did nothing\n");
          return mod;
        }
      } else {
        TORCHTRT_CHECK(
            conversion::VerifyConverterSupportForBlock(g->block()),
            "Not all operations in graph are supported by the compiler");
        // TODO find the right
        auto engine = conversion::ConvertBlockToEngine(g->block(), cfg.convert_info, static_params);
        AddEngineToGraph(new_mod, new_g, engine, cuda_device);
      }
      auto new_method = new_mod._ivalue()->compilation_unit()->create_function(method.name(), new_g);
      auto schema = util::GenerateGraphSchema(new_method->name(), new_g);
      new_mod.type()->addMethod(new_method);
      new_method->setSchema(schema);
    }
  }
  return new_mod;
}

torch::jit::script::Module EmbedEngineInNewModule(const std::string& engine, runtime::CudaDevice cuda_device) {
  std::ostringstream engine_id;
  engine_id << reinterpret_cast<const int*>(&engine);
  torch::jit::script::Module new_mod("tensorrt_engine_mod_" + engine_id.str());
  auto new_g = std::make_shared<torch::jit::Graph>();
  AddEngineToGraph(new_mod, new_g, engine, cuda_device);
  auto new_method = new_mod._ivalue()->compilation_unit()->create_function("forward", new_g);
  auto schema = util::GenerateGraphSchema(new_method->name(), new_g);
  new_mod.type()->addMethod(new_method);
  new_method->setSchema(schema);

  return new_mod;
}

void set_device(const int gpu_id) {
  TORCHTRT_ASSERT(cudaSetDevice(gpu_id) == cudaSuccess, "Unable to set CUDA device: " << gpu_id);
}

} // namespace core
} // namespace torch_tensorrt
