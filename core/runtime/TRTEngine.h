#pragma once
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "ATen/core/function_schema.h"
#include "ATen/cuda/CUDAGraph.h"
#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/custom_class.h"

#include "core/runtime/TRTEngineProfiler.h"
#include "core/util/prelude.h"

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
    std::tuple<std::string, std::string>, // serialized metadata
    std::tuple<std::string, std::string>>; // Platform

struct TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  std::shared_ptr<nvinfer1::IRuntime> rt;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
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
      const std::string& serialized_metadata = "");

  TRTEngine(std::vector<std::string> serialized_info);

  TRTEngine(
      const std::string& mod_name,
      const std::string& serialized_engine,
      const RTDevice& cuda_device,
      const std::vector<std::string>& in_binding_names,
      const std::vector<std::string>& out_binding_names,
      const Platform& target_platform = get_current_platform(),
      bool hardware_compatible = false,
      const std::string& serialized_metadata = "");

  TRTEngine& operator=(const TRTEngine& other);
  std::string to_str() const;
  static void verify_serialization_fmt(const std::vector<std::string>& serialized_info);
  void enable_profiling();
  void disable_profiling();
  std::string get_engine_layer_info();

  void dump_engine_layer_info_to_file(const std::string& path);
  void dump_engine_layer_info();
  int64_t get_device_memory_budget();
  bool set_device_memory_budget(int64_t budget);
  int64_t get_streamable_device_memory_budget();
  int64_t get_automatic_device_memory_budget();
  std::vector<at::Tensor> infer_outputs(std::vector<std::vector<int64_t>> input_shapes);
  void set_pre_allocated_outputs(bool enable);
  friend std::ostream& operator<<(std::ostream& os, const TRTEngine& engine);
  static const char BINDING_DELIM = '%';

  // Serde re-export functionality
  FlattenedState __obj_flatten__();
  std::vector<std::string> serialize();

  // CUDAGraph-Related Functionality
  at::cuda::CUDAGraph cudagraph = {};
  at::cuda::CUDAStream engine_stream = c10::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStream caller_stream = c10::cuda::getDefaultCUDAStream();
  std::vector<at::Tensor> input_buffers = {};
  std::vector<at::Tensor> output_buffers = {};
  std::string shape_key = "None";
  bool prev_cudagraphs_enabled = false;
  bool use_pre_allocated_outputs = true;
  std::vector<at::Tensor> pre_allocated_outputs;

  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);

  void set_profiling_paths();
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
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
