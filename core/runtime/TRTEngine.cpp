#include <algorithm>
#include <filesystem>
#include <utility>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/cuda.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#ifdef ENABLE_TRT_NCCL_COLLECTIVES
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/NCCLUtils.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp"
#endif

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {
// TensorRT marks unspecified dimensions in dynamic-shape engines with -1.
constexpr int32_t kDynamicDim = -1;

// Returns true iff any of the listed input bindings (including shape tensors) has a
// dynamic dimension.
[[nodiscard]] bool engine_has_dynamic_inputs(
    nvinfer1::ICudaEngine* cuda_engine,
    std::vector<std::string> const& in_binding_names) {
  TORCHTRT_CHECK(cuda_engine != nullptr, "engine_has_dynamic_inputs requires a live ICudaEngine");
  return std::any_of(std::begin(in_binding_names), std::cend(in_binding_names), [cuda_engine](std::string const& name) {
    auto const dims = cuda_engine->getTensorShape(name.c_str());
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t d) { return d == kDynamicDim; });
  });
}
} // namespace

std::string slugify(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  return s;
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

void TRTEngine::clear_active_input_tensors() {
  active_input_tensors.clear();
  active_shape_tensor_values.clear();
}

void TRTEngine::reset_active_input_tensors() {
  clear_active_input_tensors();
  active_input_tensors.resize(num_io.first);
}

void TRTEngine::record_active_input_tensor_stream_usage(const c10::cuda::CUDAStream& stream) {
  for (const auto& input : active_input_tensors) {
    if (input.defined() && input.is_cuda() && input.has_storage() && input.numel() > 0) {
      c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
    }
  }
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
    const ResourceAllocationStrategy resource_allocation_strategy,
    RuntimeSettings runtime_settings)
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
          resource_allocation_strategy,
          std::move(runtime_settings)) {}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info)
    : TRTEngine(
          serialized_info[NAME_IDX],
          serialized_info[ENGINE_IDX],
          RTDevice(serialized_info[DEVICE_IDX]),
          split_serialized_binding_names(serialized_info[INPUT_BINDING_NAMES_IDX]),
          split_serialized_binding_names(serialized_info[OUTPUT_BINDING_NAMES_IDX]),
          Platform(serialized_info[TARGET_PLATFORM_IDX]),
          static_cast<bool>(std::stoi(serialized_info[HW_COMPATIBLE_IDX])),
          static_cast<bool>(std::stoi(serialized_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])),
          serialized_info[SERIALIZED_METADATA_IDX],
          (static_cast<bool>(std::stoi(serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX]))
               ? ResourceAllocationStrategy::kDynamic
               : ResourceAllocationStrategy::kStatic),
          RuntimeSettings{}) {
  // Single visible marker that this engine was instantiated through the C++ runtime
  // entry point (i.e. torch.classes.tensorrt.Engine), distinguishing it from the Python
  // TRTEngine path. Tests look for this string in captured stderr to verify the
  // expected backend was exercised.
  LOG_INFO("[torch-TensorRT C++ runtime] TRTEngine constructed from serialized info");
  this->requires_native_multidevice = std::stoi(serialized_info[REQUIRES_NATIVE_MULTIDEVICE_IDX]);
  if (this->requires_native_multidevice) {
    LOG_INFO("Loaded distributed TRT engine (contains NCCL collectives); NCCL comm will be bound on first execution");
  }
}

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
    const ResourceAllocationStrategy resource_allocation_strategy,
    RuntimeSettings runtime_settings) {
  this->runtime_cfg = TRTRuntimeConfig(std::move(runtime_settings));
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
  default_stream = c10::cuda::getDefaultCUDAStream(device_info.id);
  owned_pool_stream = c10::cuda::getStreamFromPool(false, device_info.id);

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
  // ``exec_ctx_`` is created lazily on first ``exec_ctx()`` read so that any
  // JIT compilations can occur after all runtime settings are provided.

  // Pre-allocate placeholder for empty tensors (TensorRT requires non-null addresses)
  cudaMalloc(&empty_tensor_placeholder, 1);

  runtime_states.old_cudagraphs = CUDAGRAPHS_MODE;
  runtime_states.old_pre_allocated_outputs = false;
  runtime_states.context_changed = false;

  if (_in_binding_names.size() == 0 && _out_binding_names.size() == 0) {
    TensorRTBindingNames binding_names;
    TORCHTRT_CHECK(
        infer_engine_binding_names(*cuda_engine, binding_names),
        "Unable to determine binding indices from TensorRT engine binding names"
            << "\nEnsure module was compiled with Torch-TensorRT.ts or follows Torch-TensorRT Runtime conventions");

    num_io = std::make_pair(binding_names.input_names.size(), binding_names.output_names.size());
    in_binding_map = std::move(binding_names.input_map);
    out_binding_map = std::move(binding_names.output_map);
    in_binding_names = std::move(binding_names.input_names);
    out_binding_names = std::move(binding_names.output_names);
    cudagraph_input_staging_buffers.resize(num_io.first);
    cudagraph_output_staging_buffers.resize(num_io.second);
  } else {
    uint64_t inputs_size = _in_binding_names.size();
    in_binding_names.resize(inputs_size);
    cudagraph_input_staging_buffers.resize(inputs_size);
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
    cudagraph_output_staging_buffers.resize(outputs);
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

  has_dynamic_inputs = engine_has_dynamic_inputs(cuda_engine.get(), in_binding_names);

#ifndef NDEBUG
  this->enable_profiling();
#endif
  LOG_DEBUG(*this);

#ifdef ENABLE_TRT_NCCL_COLLECTIVES
  // Distributed engines must have a bound communicator on the IExecutionContext
  // before the first collective; bind here. ``bind_nccl_comm`` materializes
  // ``exec_ctx_`` lazily via the ``exec_ctx()`` getter if it isn't built yet.
  // For non-distributed engines we leave ``exec_ctx_`` null so the first
  // ``execute_engine`` (typically right after the Python settings dispatch)
  // is the single TRT context-create site.
  if (this->requires_native_multidevice) {
    bind_nccl_comm();
  }
#endif
}

TRTEngine::~TRTEngine() {
  // Disk persistence for runtime caches is owned by the Python side
  // (`RuntimeCacheHandle.save()` invoked from the runtime_cache CM or the engine
  // wrapper). The C++ side just lets refcounts drop.
  trt_engine_profiler.reset();
  exec_ctx_.reset();
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
  // Drop the profiler-attached context; next execute lazily creates a fresh
  // one with no profiler.
  invalidate_exec_ctx();
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
  exec_ctx()->setProfiler(trt_engine_profiler.get());
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
  // Lazy-create via the getter -- callers can hit this before the first
  // execute_engine.
  auto* ctx = exec_ctx();
  std::vector<at::Tensor> outputs;
  TORCHTRT_CHECK(
      (in_binding_names.size() == input_shapes.size()),
      "The number of input shapes provided doesn't match with the number of input names registered.");
  // Set all input shapes
  for (size_t i = 0; i < input_shapes.size(); i++) {
    ctx->setInputShape(in_binding_names[i].c_str(), core::util::toDims(input_shapes[i]));
  }
  for (size_t i = 0; i < out_binding_names.size(); i++) {
    auto output_shape = core::util::toVec(ctx->getTensorShape(out_binding_names[i].c_str()));
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
  // Weight-streaming budget cannot be modified while a context is live; drop it.
  invalidate_exec_ctx();
  if (profile_execution) {
    trt_engine_profiler.reset();
  }
  bool result = cuda_engine->setWeightStreamingBudgetV2(budget);
  // Eagerly rebuild if the user had profiling on (so the profiler is attached
  // before they query it); otherwise leave lazy.
  if (profile_execution) {
    enable_profiling();
  }
#ifdef ENABLE_TRT_NCCL_COLLECTIVES
  // Context was invalidated -- re-bind the NCCL communicator if this is a
  // distributed engine that has already been set up. ``bind_nccl_comm``
  // ensures the context before binding via the ``exec_ctx()`` getter.
  if (nccl_initialized) {
    bind_nccl_comm();
  }
#endif
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
  // Shape/dtype queries require a live IExecutionContext. Under the lazy-create
  // policy the context may be null at debug-log time (e.g. ctor-time LOG_DEBUG
  // before the first execute_engine). Probe via ``has_exec_ctx()`` so we don't
  // accidentally trigger creation just to print a debug message.
  if (!has_exec_ctx()) {
    ss << "  Inputs: <execution context not yet materialized>" << std::endl;
    ss << "  Outputs: <execution context not yet materialized>" << std::endl;
  } else {
    auto* ctx = exec_ctx_.get();
    ss << "  Inputs: [" << std::endl;
    for (uint64_t i = 0; i < num_io.first; i++) {
      ss << "    id: " << i << std::endl;
      ss << "      name: " << in_binding_names[i].c_str() << std::endl;
      ss << "      shape: " << ctx->getTensorShape(in_binding_names[i].c_str()) << std::endl;
      ss << "      dtype: "
         << util::TRTDataTypeToScalarType(ctx->getEngine().getTensorDataType(in_binding_names[i].c_str()))
         << std::endl;
    }
    ss << "  ]" << std::endl;
    ss << "  Outputs: [" << std::endl;
    for (uint64_t o = 0; o < num_io.second; o++) {
      ss << "    id: " << o << std::endl;
      ss << "      name: " << out_binding_names[o].c_str() << std::endl;
      ss << "      shape: " << ctx->getTensorShape(out_binding_names[o].c_str()) << std::endl;
      ss << "      dtype: "
         << util::TRTDataTypeToScalarType(
                ctx->getEngine().getTensorDataType(out_binding_names[o].c_str()))
         << std::endl;
    }
    ss << "  ]" << std::endl;
  }
  ss << "  Device: " << device_info << std::endl;
  ss << "  Hardware Compatibility: " << (hardware_compatible ? "Enabled" : "Disabled") << std::endl;
  ss << "  Target Platform: " << target_platform << std::endl;
  ss << "  Resource Allocation Strategy: " << (resource_allocation_strategy == ResourceAllocationStrategy::kDynamic ? "Dynamic" : "Static") << std::endl;
  ss << "  Multi-Device Engine: " << (requires_native_multidevice) << std::endl;
  ss << runtime_cfg.settings().to_str();
  // clang-format on
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TRTEngine& engine) {
  os << engine.to_str();
  return os;
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
      std::tuple("resource_allocation_strategy", serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX]),
      std::tuple("requires_native_multidevice", serialized_info[REQUIRES_NATIVE_MULTIDEVICE_IDX]));
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
  serialized_info[REQUIRES_NATIVE_MULTIDEVICE_IDX] = this->requires_native_multidevice ? "1" : "0";
  // rank/world_size are runtime facts (may differ at load time); not serialized.
  // RuntimeSettings are intentionally NOT serialized: they're per-engine, in-memory
  // initialization values, not part of the engine's identity.

  return serialized_info;
}

void TRTEngine::reset_captured_graph() {
  cudagraph.reset();
}

void TRTEngine::set_resource_allocation_strategy(TRTEngine::ResourceAllocationStrategy new_strategy) {
  if (new_strategy != this->resource_allocation_strategy) {
    this->resource_allocation_strategy = new_strategy;
    LOG_DEBUG(
        "Setting resource allocation strategy to "
        << (this->resource_allocation_strategy == TRTEngine::ResourceAllocationStrategy::kDynamic ? "dynamic"
                                                                                                  : "static"));
    invalidate_exec_ctx();
  }
}

#ifdef ENABLE_TRT_NCCL_COLLECTIVES
bool TRTEngine::bind_nccl_comm() {
  // When group_name is empty (e.g. engine loaded from a serialized
  // ExportedProgram where the Python TorchTensorRTModule wrapper was
  // inlined and set_group_name() was never called), auto-resolve the
  // process group from the c10d registry.  PyTorch assigns sequential
  // numeric names ("0", "1", ...) to process groups; probe until we
  // find one with an NCCL backend.
  if (this->group_name.empty() && this->requires_native_multidevice) {
    // PyTorch assigns sequential numeric names ("0", "1", ...) to process
    // groups.  Collect every group that has an NCCL backend; we can only
    // auto-resolve when there is exactly one — if there are several (TP+DP,
    // Megatron 4-D parallelism, etc.) we cannot know which group this engine
    // belongs to and the caller must pin it explicitly.
    std::vector<std::string> nccl_groups;
    for (int i = 0; i < 20; ++i) {
      auto candidate = std::to_string(i);
      auto probe = c10d::resolve_process_group(candidate);
      if (probe != nullptr && probe->getBackendType() == c10d::ProcessGroup::BackendType::NCCL) {
        nccl_groups.push_back(candidate);
      }
    }

    if (nccl_groups.size() == 1) {
      this->group_name = nccl_groups[0];
      LOG_INFO("Auto-resolved distributed group name to '" << this->group_name << "'");
    } else if (nccl_groups.size() > 1) {
      std::string names;
      for (const auto& n : nccl_groups) {
        if (!names.empty())
          names += ", ";
        names += "'" + n + "'";
      }
      LOG_WARNING(
          "This TRT engine requires NCCL but multiple NCCL process groups are registered ("
          << names
          << "). Cannot auto-select a group — NCCL bind deferred. "
             "Use the recommended workflow: "
             "with torch_tensorrt.distributed.distributed_context(group, model) as m: m(inp)");
    } else {
      LOG_WARNING(
          "This TRT engine requires NCCL (requires_native_multidevice=true) but no NCCL process group "
          "was found in the c10d registry. Ensure dist.init_process_group(backend='nccl') "
          "has been called before loading the engine. You can also set the group name "
          "manually via: torch_tensorrt.distributed.distributed_context(group, model)");
    }
  }

  // Soft-return when the process group isn't available yet (e.g. at engine
  // construction time when the caller hasn't called dist.init_process_group()).
  auto pg = c10d::resolve_process_group(this->group_name);
  if (pg == nullptr) {
    LOG_DEBUG("ProcessGroup '" << this->group_name << "' not yet registered in c10d; NCCL bind deferred.");
    return false;
  }

  this->rank = pg->getRank();
  this->world_size = pg->getSize();

  auto backend = pg->getBackend(c10d::ProcessGroup::BackendType::NCCL);
  TORCHTRT_CHECK(backend != nullptr, "ProcessGroup '" << this->group_name << "' has no NCCL backend");

  auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
  TORCHTRT_CHECK(nccl_pg != nullptr, "Backend is not ProcessGroupNCCL");

  at::cuda::set_device(this->device_info.id);

  int64_t comm_ptr = nccl_pg->getCommPtr();
  // Soft-return when NCCL hasn't run a collective yet.  The communicator is
  // created lazily by PyTorch on the first collective — callers should ensure
  // at least one collective (e.g. dist.barrier()) has been issued before the
  // first TRT forward pass.
  if (comm_ptr == 0) {
    LOG_DEBUG(
        "NCCL communicator not yet initialized for device " << this->device_info.id
                                                            << "; NCCL bind deferred until first execute_engine call.");
    return false;
  }

  // Distributed engines must hold a live IExecutionContext at bind time.
  // The ``exec_ctx()`` getter materializes it lazily on first call.
  exec_ctx()->setCommunicator(reinterpret_cast<void*>(comm_ptr));
  this->nccl_initialized = true;
  LOG_INFO("NCCL comm bound (rank=" << this->rank << ", device=" << this->device_info.id << ")");
  return true;
}

void TRTEngine::release_nccl_comm() {
  if (!this->nccl_initialized) {
    return;
  }
  LOG_INFO("Releasing NCCL communicator from engine '" << this->name << "'");
  torch::cuda::synchronize(device_info.id);
  invalidate_exec_ctx();
  // Eagerly rebuild so the engine returns to a "context-live, no NCCL" state
  // (callers may immediately query the context for shape/dtype info post-release).
  (void)exec_ctx();
  this->nccl_initialized = false;
  LOG_INFO("NCCL communicator released from engine '" << this->name << "'");
}
#endif // ENABLE_TRT_NCCL_COLLECTIVES

bool TRTEngine::is_monolithic_capturable(cudaStream_t stream) const {
  // Probe via the raw backing pointer -- this is a read-only check, we don't
  // want to materialize the context just to ask whether capture is feasible.
  return runtime_cfg.is_monolithic_capturable(has_dynamic_inputs, exec_ctx_.get(), stream);
}

void TRTEngine::disable_rtx_native_cudagraphs() {
#ifdef TRT_MAJOR_RTX
  if (runtime_cfg.settings().cuda_graph_strategy == CudaGraphStrategy::kDISABLED) {
    return;
  }
  LOG_WARNING(
      "Outer CUDA stream capture detected; disabling TensorRT-RTX native CUDA graph strategy on engine "
      << name << " for the remainder of its lifetime.");
  RuntimeSettings new_settings = runtime_cfg.settings();
  new_settings.cuda_graph_strategy = CudaGraphStrategy::kDISABLED;
  (void)runtime_settings(std::move(new_settings));
#endif
}

bool TRTEngine::runtime_settings(RuntimeSettings new_settings) {
  if (!runtime_cfg.settings(std::move(new_settings))) {
    return false;
  }
  // Lazy: drop the live context, but do NOT eagerly recreate. The next user
  // (typically the next ``execute_engine`` call) will lazy-create with the
  // new settings via the ``exec_ctx()`` getter. This collapses the historical
  // "ctor-create-with-defaults + dispatch-recreate-with-settings" pair on the
  // Python ``setup_engine`` cpp branch into a single create.
  invalidate_exec_ctx();
#ifdef ENABLE_TRT_NCCL_COLLECTIVES
  // The communicator was bound onto the IExecutionContext we just dropped, so
  // the next ``execute_engine`` must re-bind via ``bind_nccl_comm()``.
  nccl_initialized = false;
#endif
  // Existing recreate sites set runtime_states.context_changed for cudagraph
  // re-record; do the same here so a settings flip inside an active CM forces
  // the next enqueue to re-record any captured graph.
  runtime_states.context_changed = true;
  return true;
}

nvinfer1::IExecutionContext* TRTEngine::exec_ctx() {
  if (exec_ctx_ == nullptr) {
    recreate_execution_context();
  }
  return exec_ctx_.get();
}

void TRTEngine::invalidate_exec_ctx() noexcept {
  exec_ctx_.reset();
}

bool TRTEngine::has_exec_ctx() const noexcept {
  return exec_ctx_ != nullptr;
}

void TRTEngine::recreate_execution_context() {
  const auto allocation_strategy = resource_allocation_strategy == ResourceAllocationStrategy::kDynamic
      ? nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED
      : nvinfer1::ExecutionContextAllocationStrategy::kSTATIC;
  exec_ctx_ = runtime_cfg.create_execution_context(cuda_engine.get(), allocation_strategy);
  TORCHTRT_CHECK(exec_ctx_.get() != nullptr, "Unable to (re)create TensorRT execution context");
  ++num_execution_contexts_created_;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
