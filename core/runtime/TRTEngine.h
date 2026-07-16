#pragma once
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "ATen/core/function_schema.h"
#include "ATen/cuda/CUDAGraph.h"
#include "NvInfer.h"
#include "NvInferVersion.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/custom_class.h"

#include "core/runtime/RuntimeSettings.h"
#include "core/runtime/TRTEngineProfiler.h"
#include "core/runtime/TRTRuntimeConfig.h"
#include "core/runtime/TensorRTBindingNames.h"
#include "core/util/prelude.h"

// TensorRT 10.16+ has native NCCL collective support via IExecutionContext::setCommunicator()
#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 16)
#define TRT_HAS_NATIVE_NCCL 1
#endif

// Full TRT NCCL collectives support requires both:
// 1. PyTorch built with NCCL (USE_C10D_NCCL defined via Bazel)
// 2. TensorRT 10.16+ (TRT_HAS_NATIVE_NCCL defined above)
#if defined(USE_C10D_NCCL) && defined(TRT_HAS_NATIVE_NCCL)
#define ENABLE_TRT_NCCL_COLLECTIVES 1
#endif

namespace torch_tensorrt {

namespace core {
namespace runtime {

using FlattenedState = std::tuple<
    std::tuple<std::string, std::string>, // ABI_VERSION
    std::tuple<std::string, std::string>, // name
    std::tuple<std::string, std::string>, // device
    std::tuple<std::string, std::string>, // engine
    std::tuple<std::string, std::string>, // input binding names
    std::tuple<std::string, std::string>, // output binding names
    std::tuple<std::string, std::string>, // HW compatibility
    std::tuple<std::string, std::string>, // requires_output_allocator
    std::tuple<std::string, std::string>, // serialized metadata
    std::tuple<std::string, std::string>, // Platform
    std::tuple<std::string, std::string>, // Resource Allocation Strategy
    std::tuple<std::string, std::string> // requires_native_multidevice
    >;

struct TorchTRTRuntimeStates {
  // Indicates whether CUDAGraphs were enabled in the previous execute_engine
  bool old_cudagraphs;
  // Indicates whether pre-allocated output was enabled in the previous execute_engine
  bool old_pre_allocated_outputs;
  // Indicates whether context has changed
  bool context_changed;

  // Evaluates whether certain conditions are met to enable CUDA Graph recording/reset or to reuse pre-allocated outputs
  // based on the current and previous states, as well as input shape has changed
  std::tuple<bool, bool, bool> set_runtime_states(
      bool new_cudagraphs,
      bool new_pre_allocated_output,
      bool shape_changed) {
    bool need_cudagraphs_record = false;
    bool can_use_pre_allocated_outputs = false;
    bool need_cudagraphs_reset = false;

    // Cudagraphs record is required if cudagraphs_enabled is switched to True regardless of shape change
    if (new_cudagraphs && (!old_cudagraphs || shape_changed || context_changed)) {
      need_cudagraphs_record = true;
    }
    // Pre-allocated output can be used when previous and current state are true without shape change
    if (old_pre_allocated_outputs && new_pre_allocated_output && !shape_changed) {
      can_use_pre_allocated_outputs = true;
    }
    if (!new_cudagraphs || shape_changed || context_changed) {
      need_cudagraphs_reset = true;
    }

    old_cudagraphs = new_cudagraphs;
    old_pre_allocated_outputs = new_pre_allocated_output;
    // Reset flag
    context_changed = false;

    return {need_cudagraphs_record, can_use_pre_allocated_outputs, need_cudagraphs_reset};
  }
};

class DynamicOutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  DynamicOutputAllocator(const std::unordered_map<std::string, at::ScalarType>& output_dtypes);

  void* reallocateOutputAsync(
      char const* tensorName,
      void* currentMemory,
      uint64_t size,
      uint64_t alignment,
      cudaStream_t stream) override;

  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

  const std::unordered_map<std::string, at::Tensor>& getBuffers() const {
    return buffers;
  }

  const std::unordered_map<std::string, nvinfer1::Dims>& getShapes() const {
    return shapes;
  }

 private:
  std::unordered_map<std::string, at::ScalarType> dtypes;
  std::unordered_map<std::string, at::Tensor> buffers;
  std::unordered_map<std::string, nvinfer1::Dims> shapes;
};

struct TRTEngine : torch::CustomClassHolder {
  // Resource Allocation Strategy
  typedef enum { kStatic = 0, kDynamic } ResourceAllocationStrategy;
  // Each engine needs it's own runtime object
  std::shared_ptr<nvinfer1::IRuntime> rt;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  std::pair<uint64_t, uint64_t> num_io;
  bool output_tensors_are_unowned = false;
  std::string name;
  RTDevice device_info;

  std::string profile_path_prefix = std::filesystem::temp_directory_path().string();

  std::unordered_map<uint64_t, uint64_t> in_binding_map = {}; // TRT IDX -> PYT IDX
  std::unordered_map<uint64_t, uint64_t> out_binding_map = {}; // TRT IDX -> PYT IDX

  std::vector<std::string> in_binding_names = {}; // ITO: PYT IDX
  std::vector<std::string> out_binding_names = {}; // ITO: PYT IDX

  bool hardware_compatible = false; // Whether the engine was compiled in hardware compatible mode
  std::string serialized_metadata; // This is a base64 encoded pkl object used to store metadata such as settings used
                                   // in compilation
  Platform target_platform;

  ~TRTEngine();
  TRTEngine(
      const std::string& serialized_engine,
      const RTDevice& cuda_device,
      const std::vector<std::string>& in_binding_names,
      const std::vector<std::string>& out_binding_names,
      const Platform& target_platform = get_current_platform(),
      bool hardware_compatible = false,
      bool requires_output_allocator = false,
      const std::string& serialized_metadata = "",
      const TRTEngine::ResourceAllocationStrategy resource_allocation_strategy =
          TRTEngine::ResourceAllocationStrategy::kStatic,
      RuntimeSettings runtime_settings = RuntimeSettings{});

  TRTEngine(std::vector<std::string> serialized_info);

  TRTEngine(
      const std::string& mod_name,
      const std::string& serialized_engine,
      const RTDevice& cuda_device,
      const std::vector<std::string>& in_binding_names,
      const std::vector<std::string>& out_binding_names,
      const Platform& target_platform = get_current_platform(),
      bool hardware_compatible = false,
      bool requires_output_allocator = false,
      const std::string& serialized_metadata = "",
      const TRTEngine::ResourceAllocationStrategy resource_allocation_strategy =
          TRTEngine::ResourceAllocationStrategy::kStatic,
      RuntimeSettings runtime_settings = RuntimeSettings{});

  std::string to_str() const;
  static void verify_serialization_fmt(const std::vector<std::string>& serialized_info);
  void enable_profiling();
  void set_profile_format(std::string profile_format);
  void disable_profiling();
  std::string get_engine_layer_info();
  std::string get_serialized_metadata();

  void dump_engine_layer_info_to_file(const std::string& path);
  void dump_engine_layer_info();
  int64_t get_device_memory_budget();
  bool set_device_memory_budget(int64_t budget);
  int64_t get_streamable_device_memory_budget();
  int64_t get_automatic_device_memory_budget();
  std::vector<at::Tensor> infer_outputs(std::vector<std::vector<int64_t>> input_shapes);
  void set_output_tensors_as_unowned(bool enable);
  bool are_output_tensors_unowned();
  void clear_active_input_tensors();
  void reset_active_input_tensors();
  // Mark active input tensor allocations as used by a CUDA stream so the CUDA
  // caching allocator does not recycle their storage while that stream may still
  // access it.
  void record_active_input_tensor_stream_usage(const c10::cuda::CUDAStream& stream);
  TorchTRTRuntimeStates runtime_states;
  friend std::ostream& operator<<(std::ostream& os, const TRTEngine& engine);
  static constexpr char BINDING_DELIM = kBindingNameDelimiter;

  // Serde re-export functionality
  FlattenedState __obj_flatten__();
  std::vector<std::string> serialize();

  // CUDAGraph-Related Functionality
  at::cuda::CUDAGraph cudagraph = {};
  at::cuda::CUDAStream engine_stream = c10::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStream caller_stream = c10::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStream default_stream = c10::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStream owned_pool_stream = c10::cuda::getDefaultCUDAStream();
  std::vector<at::Tensor> cudagraph_input_staging_buffers = {};
  std::vector<at::Tensor> cudagraph_output_staging_buffers = {};

  // Per-call formatted input buffers. In standard mode these are bound
  // directly to TRT; in CUDA graph mode they are async-copy sources for
  // persistent CUDA graph input staging buffers.
  struct InputBindingInfo {
    std::string name;
    at::ScalarType expected_type;
    bool is_shape_tensor;
  };
  std::vector<InputBindingInfo> input_binding_infos = {};
  std::vector<at::Tensor> active_input_tensors = {};
  std::list<std::vector<int64_t>> active_shape_tensor_values = {};

  std::string shape_key = "None";
  bool use_pre_allocated_outputs = false;
  std::vector<at::Tensor> pre_allocated_outputs;

  // --- Multiple optimization profiles ---
  // State and helpers mirror the Python runtime (TRTEngine in _TRTEngine.py) so
  // the C++ and Python runtimes are interchangeable: the same attribute and
  // method names are exposed via torchbind in register_jit_hooks.cpp
  // (``num_optimization_profiles``, ``_active_profile_index``,
  // ``_auto_select_profiles``, ``set_active_profile``). Index validation lives
  // in the runtime-agnostic TorchTensorRTModule.resolve_profile_index.
  int64_t num_optimization_profiles = 1; // cuda_engine->getNbOptimizationProfiles()
  int64_t active_profile_index = 0; // profile currently loaded in exec_ctx
  bool auto_select_profiles = false; // opt-in shape-based selection (per call)
  // A single input dimension whose extent varies across or within optimization
  // profiles, paired with its [min, max] range for each profile index. Dims that
  // are a fixed identical extent in every profile are NOT stored, so selection
  // only inspects dims that can actually distinguish profiles.
  struct DynamicProfileDim {
    int32_t dim_index;
    std::vector<std::pair<int64_t, int64_t>> profile_ranges; // indexed by profile index
  };
  // Indexed by input binding position (parallel to ``in_binding_names``, so it
  // reuses the same positional input -> binding mapping the rest of the runtime
  // relies on). Each entry holds only that input's dynamic dims; shape-inference
  // IO and all-static inputs get an empty list. Built once in
  // setup_optimization_profiles so auto-selection does no per-call string
  // lookups.
  std::vector<std::vector<DynamicProfileDim>> profile_dynamic_dims;

  // Cache profile count + per-profile dim ranges purely from the TRT API
  // (getNbOptimizationProfiles / getProfileShape) so selection works for any
  // loaded engine with no extra serialized metadata.
  void setup_input_binding_infos();
  void setup_optimization_profiles();
  // Switch the active TRT optimization profile (idempotent). Manual-pin /
  // torchbind entry point: runs outside execute_engine's stream setup, so it
  // resolves the current stream and fully synchronizes (rare, not perf-critical).
  void set_active_profile(int64_t profile_index);
  // Core switch issued on ``stream`` with no host synchronize: the caller must
  // guarantee a happens-before to the enqueue (e.g. issue on the enqueue stream).
  // Used by auto-selection, which switches on engine_stream before enqueueV3.
  void set_active_profile_with_stream(int64_t profile_index, const c10::cuda::CUDAStream& stream);
  // True if every input's dynamic dims fit profile ``profile_index``'s [min, max]
  // ranges (positional inputs[i] <-> in_binding_names[i]; static / shape-inference
  // IO inputs are skipped).
  bool profile_fits(int64_t profile_index, const std::vector<at::Tensor>& inputs) const;
  // Select and activate the optimization profile for ``inputs`` and return its
  // index. Keeps the currently active profile when it still fits (no switch, no
  // thrashing); otherwise switches to the first fitting profile. Owns the switch
  // so callers need no follow-up. Called internally from the execute_engine run
  // paths (guarded by num_optimization_profiles > 1 && auto_select_profiles);
  // manual pins are applied eagerly via set_active_profile.
  int64_t auto_select_profile(const std::vector<at::Tensor>& inputs);

  // Single placeholder buffer for empty tensor inputs (allocated once, reused)
  void* empty_tensor_placeholder = nullptr;

  // Output Allocator-Related Functionality
  bool requires_output_allocator = false; // engine requires output allocator
  bool use_output_allocator_outputs = false; // users specify to use output allocator
  std::shared_ptr<DynamicOutputAllocator> output_allocator;

  // Member variables for distributed inference
  bool requires_native_multidevice = false; // compile-time flag: engine contains NCCL collectives
  int64_t rank = -1; // populated at runtime by setup_nccl_comm()
  int64_t world_size = -1; // populated at runtime by setup_nccl_comm()
  std::string group_name = ""; // c10d registry name; "" = default world group

#ifdef ENABLE_TRT_NCCL_COLLECTIVES
  const bool _native_nccl_support = true; // Support value that is mostly here to back the torchbind hooks
  bool nccl_initialized = false; // guards lazy one-shot NCCL setup in execute_engine

  // Resolve ProcessGroup via group_name, fetch the NCCL comm from PyTorch,
  // and bind it to exec_ctx.  Returns true on success.  Returns false (without
  // throwing) when the process group or NCCL communicator is not yet available
  // so callers can retry later.  Throws on hard misconfiguration (wrong backend).
  bool bind_nccl_comm();

  // Detach the NCCL communicator from the execution context by recreating it.
  // After this call the process group can be safely destroyed without causing a
  // use-after-free in the TRT engine destructor.  If the engine is used again
  // later (with a new PG), execute_engine() will see nccl_initialized=false
  // and re-bind automatically.
  void release_nccl_comm();
#else
  const bool _native_nccl_support = false;
#endif

  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);

  void set_profiling_paths();
  void reset_captured_graph();
#ifndef NDEBUG
  bool profile_execution = true;
#else
  bool profile_execution = false;
#endif
  std::string device_profile_path;
  std::string input_profile_path;
  std::string output_profile_path;
  std::string enqueue_profile_path;
  std::string trt_engine_profile_path;
  std::string cuda_graph_debug_path;
  std::mutex mu;
  std::unique_ptr<TRTEngineProfiler> trt_engine_profiler;
  ResourceAllocationStrategy resource_allocation_strategy = kStatic;
  void set_resource_allocation_strategy(ResourceAllocationStrategy new_strategy);
  ResourceAllocationStrategy get_resource_allocation_strategy();

  // Owns the canonical `RuntimeSettings` plus the live `IRuntimeConfig` derived
  // from them.
  TRTRuntimeConfig runtime_cfg;

  [[nodiscard]] RuntimeSettings const& runtime_settings() const noexcept {
    return runtime_cfg.settings();
  }

  // Setter. Returns true iff the settings actually changed -- consumers can
  // read the diff result to decide whether to invalidate dependent state. On
  // change, invalidates the live ``IRuntimeConfig`` (the next ``exec_ctx()``
  // getter call rebuilds with the new settings).
  [[nodiscard]] bool runtime_settings(RuntimeSettings new_settings);

  // Whether the engine has any input binding with a dynamic dimension. Computed
  // once during construction; used by `is_monolithic_capturable`.
  bool has_dynamic_inputs = false;

  // Monolithic-capturability check used when this engine is wrapped by an outer whole-graph
  // capture (e.g. CudaGraphsTorchTensorRTModule). Non-RTX builds always return true.
  bool is_monolithic_capturable(cudaStream_t stream) const;

  // Disable TensorRT-RTX native CUDA graph capture on this engine (one-shot, invoked when
  // an outer stream capture is detected around execute_engine). No-op on non-RTX or when
  // already disabled.
  void disable_rtx_native_cudagraphs();

  // Obtains the execution context. Materializes it lazily on first call using
  // the current settings from ``runtime_cfg`` and subsequent calls return the
  // cached instance until invalidated.
  // Returns a raw pointer owned by the internal ``shared_ptr``; do not store
  // across an invalidate. Returned pointer is never null (the underlying
  // factory throws if creation fails).
  nvinfer1::IExecutionContext* exec_ctx();

  // Drop the live execution context without recreating. The next ``exec_ctx()``
  // call rebuilds from the current ``runtime_cfg`` settings.
  void invalidate_exec_ctx() noexcept;

  // True iff the execution context has been materialized. Probes WITHOUT
  // triggering creation; for tests/introspection.
  [[nodiscard]] bool has_exec_ctx() const noexcept;

  [[nodiscard]] int64_t num_execution_contexts_created() const noexcept {
    return num_execution_contexts_created_;
  }

 private:
  // Backing storage for the execution context. External code reaches it only
  // through ``exec_ctx()`` / ``invalidate_exec_ctx()`` / ``has_exec_ctx()``.
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx_;

  // Single entry point that (re)creates exec_ctx_ via runtime_cfg.create_execution_context.
  // Bumps ``num_execution_contexts_created_``. Called from ``exec_ctx()`` when
  // the cached context is null.
  void recreate_execution_context();

  int64_t num_execution_contexts_created_ = 0;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
