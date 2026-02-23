#include <algorithm>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/cuda.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string slugify(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  return s;
}

std::vector<std::string> split(const std::string& str, char delim) {
  std::vector<std::string> strings;
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    strings.push_back(str.substr(start, end - start));
  }
  return strings;
}

DynamicOutputAllocator::DynamicOutputAllocator(const std::unordered_map<std::string, at::ScalarType>& output_dtypes)
    : dtypes(output_dtypes) {}

void* DynamicOutputAllocator::reallocateOutputAsync(
    char const* tensorName,
    void* currentMemory,
    uint64_t size,
    uint64_t alignment,
    cudaStream_t stream) {
  std::vector<int64_t> shape = {static_cast<int64_t>(size)};
  auto it = buffers.find(tensorName);
  if (it == buffers.end() || it->second.sizes() != shape) {
    buffers[tensorName] = at::empty(shape, at::TensorOptions().dtype(dtypes.at(tensorName)).device(at::kCUDA));
    return buffers[tensorName].data_ptr();
  } else {
    return it->second.data_ptr();
  }
}

void DynamicOutputAllocator::notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept {
  shapes[tensorName] = dims;
}

TRTEngine::TRTEngine(
    const std::string& serialized_engine,
    const RTDevice& cuda_device,
    const std::vector<std::string>& _in_binding_names,
    const std::vector<std::string>& _out_binding_names,
    const Platform& target_platform,
    bool hardware_compatible,
    bool requires_output_allocator,
    const std::string& serialized_metadata,
    const ResourceAllocationStrategy resource_allocation_strategy)
    : TRTEngine(
          "deserialized_trt",
          serialized_engine,
          cuda_device,
          _in_binding_names,
          _out_binding_names,
          target_platform,
          hardware_compatible,
          requires_output_allocator,
          serialized_metadata,
          resource_allocation_strategy) {}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info)
    : TRTEngine(
          serialized_info[NAME_IDX],
          serialized_info[ENGINE_IDX],
          RTDevice(serialized_info[DEVICE_IDX]),
          split(serialized_info[INPUT_BINDING_NAMES_IDX], BINDING_DELIM),
          split(serialized_info[OUTPUT_BINDING_NAMES_IDX], BINDING_DELIM),
          Platform(serialized_info[TARGET_PLATFORM_IDX]),
          static_cast<bool>(std::stoi(serialized_info[HW_COMPATIBLE_IDX])),
          static_cast<bool>(std::stoi(serialized_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])),
          serialized_info[SERIALIZED_METADATA_IDX],
          (static_cast<bool>(std::stoi(serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX]))
               ? ResourceAllocationStrategy::kDynamic
               : ResourceAllocationStrategy::kStatic)) {}

TRTEngine::TRTEngine(
    const std::string& mod_name,
    const std::string& serialized_engine,
    const RTDevice& cuda_device,
    const std::vector<std::string>& _in_binding_names,
    const std::vector<std::string>& _out_binding_names,
    const Platform& target_platform,
    bool hardware_compatible,
    bool requires_output_allocator,
    const std::string& serialized_metadata,
    const ResourceAllocationStrategy resource_allocation_strategy) {
  TORCHTRT_CHECK(
      is_supported_on_current_platform(target_platform),
      "This engine was not built to run on this platform (built for: " << target_platform << ", current platform: "
                                                                       << get_current_platform() << ")");
  this->target_platform = target_platform;

  this->hardware_compatible = hardware_compatible;
  auto most_compatible_device = get_most_compatible_device(cuda_device, RTDevice(), hardware_compatible);
  TORCHTRT_CHECK(most_compatible_device, "No compatible device was found for instantiating TensorRT engine");

  this->serialized_metadata = serialized_metadata;
  this->requires_output_allocator = requires_output_allocator;
  device_info = most_compatible_device.value();
  multi_gpu_device_check();
  set_rt_device(device_info);

  rt = make_trt(nvinfer1::createInferRuntime(util::logging::get_logger()));

  name = slugify(mod_name);

  cuda_engine = make_trt(rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size()));
  TORCHTRT_CHECK((cuda_engine.get() != nullptr), "Unable to deserialize the TensorRT engine");

  if (get_streamable_device_memory_budget() > 0) {
    int64_t budget_bytes = get_automatic_device_memory_budget();
    LOG_DEBUG("Weight streaming budget set to " << budget_bytes << "B");
    cuda_engine->setWeightStreamingBudgetV2(budget_bytes);
  }

  this->resource_allocation_strategy = resource_allocation_strategy;
  LOG_DEBUG(
      "Resource allocation strategy: "
      << (this->resource_allocation_strategy == ResourceAllocationStrategy::kDynamic ? "Dynamic" : "Static"));
  if (this->resource_allocation_strategy == ResourceAllocationStrategy::kDynamic) {
    this->exec_ctx =
        make_trt(cuda_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  } else {
    this->exec_ctx = make_trt(cuda_engine->createExecutionContext());
  }
  TORCHTRT_CHECK((exec_ctx.get() != nullptr), "Unable to create TensorRT execution context");

  // Pre-allocate placeholder for empty tensors (TensorRT requires non-null addresses)
  cudaMalloc(&empty_tensor_placeholder, 1);

  runtime_states.old_cudagraphs = CUDAGRAPHS_MODE;
  runtime_states.old_pre_allocated_outputs = false;
  runtime_states.context_changed = false;

  if (_in_binding_names.size() == 0 && _out_binding_names.size() == 0) {
    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t trt_idx = 0; trt_idx < cuda_engine->getNbIOTensors(); trt_idx++) {
      std::string bind_name = cuda_engine->getIOTensorName(trt_idx);
      LOG_DEBUG("Binding name: " << bind_name);
      auto delim = bind_name.find(".");
      if (delim == std::string::npos) {
        delim = bind_name.find("_");
        TORCHTRT_CHECK(
            delim != std::string::npos,
            "Unable to determine binding index for input "
                << bind_name
                << "\nEnsure module was compiled with Torch-TensorRT.ts or follows Torch-TensorRT Runtime conventions");
      }
      std::string idx_s = bind_name.substr(delim + 1);
      uint64_t pyt_idx = static_cast<uint64_t>(std::stoi(idx_s));

      if (cuda_engine->getTensorIOMode(bind_name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
        inputs++;
        in_binding_map[trt_idx] = pyt_idx;
        LOG_DEBUG("TRT Binding index: " << trt_idx << "corresponds to PYT Input index: " << pyt_idx);
      } else {
        outputs++;
        out_binding_map[trt_idx] = pyt_idx;
        LOG_DEBUG("TRT Binding index: " << trt_idx << "corresponds to PYT Output: " << pyt_idx);
      }
    }

    num_io = std::make_pair(inputs, outputs);
    in_binding_names.resize(inputs);
    input_buffers.resize(inputs);
    out_binding_names.resize(outputs);
    output_buffers.resize(outputs);
    for (int64_t x = 0; x < cuda_engine->getNbIOTensors(); x++) {
      std::string bind_name = cuda_engine->getIOTensorName(x);
      if (cuda_engine->getTensorIOMode(bind_name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
        in_binding_names[in_binding_map.at(x)] = bind_name;
      } else {
        out_binding_names[out_binding_map.at(x)] = bind_name;
      }
    }
  } else {
    uint64_t inputs_size = _in_binding_names.size();
    in_binding_names.resize(inputs_size);
    input_buffers.resize(inputs_size);
    for (uint64_t pyt_idx = 0; pyt_idx < inputs_size; pyt_idx++) {
      auto binding_name = _in_binding_names[pyt_idx];
      // Check if the binding name provided is in the list of engine's bindings
      // by iterating through nbIOTensors and verify it is an input binding
      bool is_binding = false, is_input = false;
      int32_t trt_idx;
      for (int32_t idx = 0; idx < cuda_engine->getNbIOTensors(); idx++) {
        std::string curr_bind_name = cuda_engine->getIOTensorName(idx);
        if (curr_bind_name == binding_name) {
          is_binding = true;
          trt_idx = idx;
          if (cuda_engine->getTensorIOMode(binding_name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            is_input = true;
            break;
          }
        }
      }
      TORCHTRT_CHECK(is_binding, "Could not find a TensorRT engine binding for input named " << binding_name);
      TORCHTRT_CHECK(
          is_input, "Binding " << binding_name << " specified as input but found as output in TensorRT engine");
      LOG_DEBUG(
          "Input binding name: " << binding_name << " has TensorRT binding index: " << trt_idx
                                 << ", Torch binding index: " << pyt_idx);
      in_binding_map[trt_idx] = pyt_idx;
      in_binding_names[pyt_idx] = binding_name;
    }

    uint64_t outputs = _out_binding_names.size();
    out_binding_names.resize(outputs);
    output_buffers.resize(outputs);
    for (size_t pyt_idx = 0; pyt_idx < outputs; pyt_idx++) {
      auto binding_name = _out_binding_names[pyt_idx];
      // Check if the binding name provided is in the list of engine's bindings
      // by iterating through nbIOTensors and verify it is an output binding
      bool is_binding = false, is_output = false;
      int32_t trt_idx;
      for (int32_t idx = 0; idx < cuda_engine->getNbIOTensors(); idx++) {
        std::string curr_bind_name = cuda_engine->getIOTensorName(idx);
        if (curr_bind_name == binding_name) {
          is_binding = true;
          trt_idx = idx;
          if (cuda_engine->getTensorIOMode(binding_name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) {
            is_output = true;
            break;
          }
        }
      }

      TORCHTRT_CHECK(is_binding, "Could not find a TensorRT engine binding for output named " << binding_name);
      TORCHTRT_CHECK(
          is_output, "Binding " << binding_name << " specified as output but found as input in TensorRT engine");

      LOG_DEBUG(
          "Output binding name: " << binding_name << " has TensorRT binding index: " << trt_idx
                                  << ", Torch binding index: " << inputs_size + pyt_idx);
      out_binding_map[trt_idx] = pyt_idx;
      out_binding_names[pyt_idx] = binding_name;
    }
    num_io = std::make_pair(inputs_size, outputs);
  }

#ifndef NDEBUG
  this->enable_profiling();
#endif
  LOG_DEBUG(*this);
}

TRTEngine::~TRTEngine() {
  trt_engine_profiler.reset();
  exec_ctx.reset();
  cuda_engine.reset();
  if (empty_tensor_placeholder) {
    cudaFree(empty_tensor_placeholder);
  }
  rt.reset();
}

void TRTEngine::disable_profiling() {
  torch::cuda::synchronize(device_info.id);
  profile_execution = false;
  trt_engine_profiler.reset();
  exec_ctx = make_trt(cuda_engine->createExecutionContext());
  TORCHTRT_CHECK((exec_ctx.get() != nullptr), "Unable to recreate TensorRT execution context");
}

void TRTEngine::dump_engine_layer_info_to_file(const std::string& path) {
  auto inspector = make_trt(cuda_engine->createEngineInspector());
  std::ofstream f(path);
  f << std::string(inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON));
  f.close();
  return;
}

void TRTEngine::dump_engine_layer_info() {
  std::string layer_info_file =
      std::filesystem::path{profile_path_prefix + "/" + name + "_layer_information.json"}.string();
  dump_engine_layer_info_to_file(layer_info_file);
  return;
}

void TRTEngine::enable_profiling() {
  profile_execution = true;
  trt_engine_profiler = std::make_unique<TRTEngineProfiler>(name);
  exec_ctx->setProfiler(trt_engine_profiler.get());
}

void TRTEngine::set_output_tensors_as_unowned(bool enable) {
  this->output_tensors_are_unowned = enable;
}

bool TRTEngine::are_output_tensors_unowned() {
  return this->output_tensors_are_unowned;
}

void TRTEngine::set_profile_format(std::string format) {
  if (format == "trex") {
    this->trt_engine_profiler->set_profile_format(TraceFormat::kTREX);
  } else if (format == "perfetto") {
    this->trt_engine_profiler->set_profile_format(TraceFormat::kPERFETTO);
  } else {
    TORCHTRT_THROW_ERROR("Invalid profile format: " + format);
  }
}

std::string TRTEngine::get_engine_layer_info() {
  auto inspector = make_trt(cuda_engine->createEngineInspector());
  return inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
}

std::string TRTEngine::get_serialized_metadata() {
  return this->serialized_metadata;
}

std::vector<at::Tensor> TRTEngine::infer_outputs(std::vector<std::vector<int64_t>> input_shapes) {
  std::vector<at::Tensor> outputs;
  TORCHTRT_CHECK(
      (in_binding_names.size() == input_shapes.size()),
      "The number of input shapes provided doesn't match with the number of input names registered.");
  // Set all input shapes
  for (size_t i = 0; i < input_shapes.size(); i++) {
    exec_ctx->setInputShape(in_binding_names[i].c_str(), core::util::toDims(input_shapes[i]));
  }
  for (size_t i = 0; i < out_binding_names.size(); i++) {
    auto output_shape = core::util::toVec(exec_ctx->getTensorShape(out_binding_names[i].c_str()));
    auto output_dtype =
        core::util::TRTDataTypeToScalarType(cuda_engine->getTensorDataType(out_binding_names[i].c_str()));
    auto output_tensor = torch::empty(output_shape, torch::dtype(output_dtype));
    outputs.push_back(output_tensor);
  }
  TORCHTRT_CHECK(
      (out_binding_names.size() == outputs.size()),
      "The number of output shapes inferred doesn't match with the number of output names registered.");
  return outputs;
}

void TRTEngine::set_profiling_paths() {
  device_profile_path =
      std::filesystem::path{profile_path_prefix + "/" + name + "_device_config_profile.trace"}.string();
  input_profile_path = std::filesystem::path{profile_path_prefix + "/" + name + "_input_profile.trace"}.string();
  output_profile_path = std::filesystem::path{profile_path_prefix + "/" + name + "_output_profile.trace"}.string();
  enqueue_profile_path = std::filesystem::path{profile_path_prefix + "/" + name + "_enqueue_profile.trace"}.string();
  trt_engine_profile_path =
      std::filesystem::path{profile_path_prefix + "/" + name + "_engine_execution_profile.trace"}.string();
  cuda_graph_debug_path = std::filesystem::path{profile_path_prefix + "/" + name + "_cudagraph.dot"}.string();
}

int64_t TRTEngine::get_device_memory_budget() {
  return cuda_engine->getWeightStreamingBudgetV2();
}

bool TRTEngine::set_device_memory_budget(int64_t budget) {
  // Recreating the context because weight streaming budget cannot be modified while there are active context.
  if (exec_ctx.get() != nullptr) {
    exec_ctx.reset();
  }
  if (profile_execution) {
    trt_engine_profiler.reset();
  }
  bool result = cuda_engine->setWeightStreamingBudgetV2(budget);
  exec_ctx = make_trt(cuda_engine->createExecutionContext());
  TORCHTRT_CHECK(
      (exec_ctx.get() != nullptr),
      "Unable to recreate TensorRT execution context after setting new device memory budget");
  if (profile_execution) {
    enable_profiling();
  }
  // Indicates to reevaluate the runtime settings
  runtime_states.context_changed = true;

  return result;
}

// Returns 0 if BuilderFlag::kWEIGHT_STREAMING is unset during engine building.
int64_t TRTEngine::get_streamable_device_memory_budget() {
  return cuda_engine->getStreamableWeightsSize();
}

int64_t TRTEngine::get_automatic_device_memory_budget() {
  return cuda_engine->getWeightStreamingAutomaticBudget();
}

std::string TRTEngine::to_str() const {
  // clang-format off
  std::stringstream ss;
  ss << "Torch-TensorRT TensorRT Engine:" << std::endl;
  ss << "  Name: " << name << std::endl;
  ss << "  Inputs: [" << std::endl;
  for (uint64_t i = 0; i < num_io.first; i++) {
    ss << "    id: " << i << std::endl;
    ss << "      name: " << in_binding_names[i].c_str() << std::endl;
    ss << "      shape: " << exec_ctx->getTensorShape(in_binding_names[i].c_str()) << std::endl;
    ss << "      dtype: "
       << util::TRTDataTypeToScalarType(exec_ctx->getEngine().getTensorDataType(in_binding_names[i].c_str()))
       << std::endl;
  }
  ss << "  ]" << std::endl;
  ss << "  Outputs: [" << std::endl;
  for (uint64_t o = 0; o < num_io.second; o++) {
    ss << "    id: " << o << std::endl;
    ss << "      name: " << out_binding_names[o].c_str() << std::endl;
    ss << "      shape: " << exec_ctx->getTensorShape(out_binding_names[o].c_str()) << std::endl;
    ss << "      dtype: "
       << util::TRTDataTypeToScalarType(
              exec_ctx->getEngine().getTensorDataType(out_binding_names[o].c_str()))
       << std::endl;
  }
  ss << "  ]" << std::endl;
  ss << "  Device: " << device_info << std::endl;
  ss << "  Hardware Compatibility: " << (hardware_compatible ? "Enabled" : "Disabled") << std::endl;
  ss << "  Target Platform: " << target_platform << std::endl;
  ss << "  Resource Allocation Strategy: " << (resource_allocation_strategy == ResourceAllocationStrategy::kDynamic ? "Dynamic" : "Static") << std::endl;
  // clang-format on
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TRTEngine& engine) {
  os << engine.to_str();
  return os;
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
  rt = other.rt;
  cuda_engine = other.cuda_engine;
  device_info = other.device_info;
  exec_ctx = other.exec_ctx;
  num_io = other.num_io;
  return (*this);
}

void TRTEngine::verify_serialization_fmt(const std::vector<std::string>& serialized_info) {
  TORCHTRT_CHECK(
      serialized_info.size() == SERIALIZATION_LEN,
      "Program to be deserialized targets an incompatible Torch-TensorRT ABI");
  TORCHTRT_CHECK(
      serialized_info[ABI_TARGET_IDX] == ABI_VERSION,
      "Program to be deserialized targets a different Torch-TensorRT ABI Version ("
          << serialized_info[ABI_TARGET_IDX] << ") than the Torch-TensorRT Runtime ABI Version (" << ABI_VERSION
          << ")");
}

FlattenedState TRTEngine::__obj_flatten__() {
  // This method would be called by meta kernel of this custom class and it only needs to return a tuple
  std::vector<std::string> serialized_info = this->serialize();

  return std::tuple(
      std::tuple("version", serialized_info[ABI_TARGET_IDX]),
      std::tuple("name", serialized_info[NAME_IDX]),
      std::tuple("device_info", serialized_info[DEVICE_IDX]),
      std::tuple("serialized_engine", serialized_info[ENGINE_IDX]),
      std::tuple("in_binding_names", serialized_info[INPUT_BINDING_NAMES_IDX]),
      std::tuple("out_binding_names", serialized_info[OUTPUT_BINDING_NAMES_IDX]),
      std::tuple("hardware_compatible", serialized_info[HW_COMPATIBLE_IDX]),
      std::tuple("serialized_metadata", serialized_info[SERIALIZED_METADATA_IDX]),
      std::tuple("requires_output_allocator", serialized_info[REQUIRES_OUTPUT_ALLOCATOR_IDX]),
      std::tuple("target_platform", serialized_info[TARGET_PLATFORM_IDX]),
      std::tuple("resource_allocation_strategy", serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX]));
}

std::vector<std::string> TRTEngine::serialize() {
  // Serialize TensorRT engine
  auto serialized_trt_engine = make_trt(this->cuda_engine->serialize());

  // Adding device info related meta data to the serialized file
  auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

  std::vector<std::string> serialized_info;
  serialized_info.resize(SERIALIZATION_LEN);

  serialized_info[ABI_TARGET_IDX] = ABI_VERSION;
  serialized_info[NAME_IDX] = this->name;
  serialized_info[DEVICE_IDX] = this->device_info.serialize();
  serialized_info[ENGINE_IDX] = base64_encode(trt_engine);
  serialized_info[INPUT_BINDING_NAMES_IDX] = serialize_bindings(this->in_binding_names);
  serialized_info[OUTPUT_BINDING_NAMES_IDX] = serialize_bindings(this->out_binding_names);
  serialized_info[HW_COMPATIBLE_IDX] = this->hardware_compatible ? "1" : "0";
  serialized_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = this->requires_output_allocator ? "1" : "0";
  serialized_info[SERIALIZED_METADATA_IDX] = this->serialized_metadata;
  serialized_info[TARGET_PLATFORM_IDX] = this->target_platform.serialize();
  serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX] =
      this->resource_allocation_strategy == ResourceAllocationStrategy::kDynamic ? "1" : "0";

  return serialized_info;
}

void TRTEngine::reset_captured_graph() {
  cudagraph.reset();
}

void TRTEngine::set_resource_allocation_strategy(TRTEngine::ResourceAllocationStrategy new_strategy) {
  if (new_strategy != this->resource_allocation_strategy) {
    this->resource_allocation_strategy = new_strategy;
    if (this->resource_allocation_strategy == TRTEngine::ResourceAllocationStrategy::kDynamic) {
      LOG_DEBUG("Setting resource allocation strategy to dynamic");
      this->exec_ctx =
          make_trt(cuda_engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    } else {
      LOG_DEBUG("Setting resource allocation strategy to static");
      this->exec_ctx = make_trt(cuda_engine->createExecutionContext());
    }
  }
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
