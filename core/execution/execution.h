#pragma once
#include <utility>
#include "NvInfer.h"
#include "ATen/core/function_schema.h"
#include "torch/custom_class.h"
#include "core/util/prelude.h"


namespace trtorch {
namespace core {
namespace execution {

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

    ~TRTEngine();
    TRTEngine(std::string serialized_engine);
    TRTEngine(std::string mod_name, std::string serialized_engine);
    TRTEngine& operator=(const TRTEngine& other);
    // TODO: Implement a call method
    //c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

std::vector<at::Tensor> RunCudaEngine(nvinfer1::IExecutionContext* ctx, std::pair<uint64_t, uint64_t> io, std::vector<at::Tensor>& inputs);

} // namespace execution
} // namespace core
} // namespace trtorch
