#pragma once
#include <utility>
#include "NvInfer.h"
#include "ATen/core/function_schema.h"
#include "torch/custom_class.h"


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

    TRTEngine() = default;
    TRTEngine(std::string serialized_engine);
    TRTEngine(std::string name, std::string serialized_engine);
    TRTEngine(nvinfer1::ILogger& logger, std::string name, std::string& serialized_engine);
    TRTEngine& operator=(const TRTEngine& other);
    void test();
};

} // namespace execution
} // namespace core
} // namespace trtorch
