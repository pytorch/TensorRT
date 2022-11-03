#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  std::shared_ptr<nvinfer1::IRuntime> rt;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
  std::string name;
  std::mutex mu;
  CUDADevice device_info;

  std::string execution_profile_path;
  std::string device_profile_path;
  std::string input_profile_path;
  std::string output_profile_path;
  std::string enqueue_profile_path;
  std::string profile_path = "/tmp";

  std::unordered_map<uint64_t, uint64_t> in_binding_map; // TRT IDX -> PYT IDX
  std::unordered_map<uint64_t, uint64_t> out_binding_map; // TRT IDX -> PYT IDX

  std::vector<std::string> in_binding_names; // ITO: PYT IDX
  std::vector<std::string> out_binding_names; // ITO: PYT IDX

#ifndef NDEBUG
  bool debug = true;
#else
  bool debug = false;
#endif

  ~TRTEngine() = default;
  TRTEngine(
      std::string serialized_engine,
      CUDADevice cuda_device,
      const std::vector<std::string>& in_binding_names,
      const std::vector<std::string>& out_binding_names);
  TRTEngine(std::vector<std::string> serialized_info);
  TRTEngine(
      std::string mod_name,
      std::string serialized_engine,
      CUDADevice cuda_device,
      const std::vector<std::string>& in_binding_names,
      const std::vector<std::string>& out_binding_names);
  TRTEngine& operator=(const TRTEngine& other);
  std::string to_str() const;
  void set_paths();
  friend std::ostream& operator<<(std::ostream& os, const TRTEngine& engine);
  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
