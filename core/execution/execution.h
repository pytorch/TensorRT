#pragma once
#include <utility>
#include "NvInfer.h"
#include "ATen/core/function_schema.h"

namespace trtorch {
namespace core {
namespace execution {

using EngineID = int64_t;

struct TRTEngine {
    // Each engine needs it's own runtime object
    nvinfer1::IRuntime* rt;
    nvinfer1::ICudaEngine* cuda_engine;
    nvinfer1::IExecutionContext* exec_ctx;
    std::pair<uint64_t, uint64_t> num_io;
    c10::FunctionSchema schema;
    EngineID id;

    TRTEngine();
    TRTEngine(nvinfer1::ILogger& logger, std::string& serialized_engine);
    TRTEngine& operator=(const TRTEngine& other);
};

void RegisterEngineOp(TRTEngine& engine);
uint64_t RegisterEngineFromSerializedEngine(std::string& serialized_engine);
nvinfer1::ICudaEngine* GetCudaEngine(EngineID id);
nvinfer1::IExecutionContext* GetExecCtx(EngineID id);
std::pair<uint64_t, uint64_t> GetEngineIO(EngineID id);
c10::FunctionSchema GetEngineFunctionSchema(EngineID id);
void DeregisterEngine(EngineID id);

} // namespace execution
} // namespace core
} // namespace trtorch
