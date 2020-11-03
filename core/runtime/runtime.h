#pragma once
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace trtorch {
namespace core {
namespace runtime {

using EngineID = int64_t;

struct TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  nvinfer1::IRuntime* rt;
  nvinfer1::ICudaEngine* cuda_engine;
  nvinfer1::IExecutionContext* exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
  EngineID id;
  std::string name;
  util::logging::TRTorchLogger logger;

  std::unordered_map<uint64_t, uint64_t> in_binding_map;
  std::unordered_map<uint64_t, uint64_t> out_binding_map;

  ~TRTEngine();
  TRTEngine(std::string serialized_engine);
  TRTEngine(std::string mod_name, std::string serialized_engine);
  TRTEngine& operator=(const TRTEngine& other);
  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

} // namespace runtime
} // namespace core
} // namespace trtorch
