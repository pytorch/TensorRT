#include <algorithm>

#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {

TRTEngine::TRTEngine(nvinfer1::ILogger& logger, std::string& serialized_engine) {
    rt = nvinfer1::createInferRuntime(logger);

    cuda_engine = rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size());
    // Easy way to get a unique name for each engine, maybe there is a more descriptive way (using something associated with the graph maybe)
    id = reinterpret_cast<EngineID>(cuda_engine);
    exec_ctx = cuda_engine->createExecutionContext();

    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
        if(cuda_engine->bindingIsInput(x)) {
            inputs++;
        } else {
            outputs++;
        }
    }
    num_io = std::make_pair(inputs, outputs);
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
    id = other.id;
    rt = other.rt;
    cuda_engine = other.cuda_engine;
    exec_ctx = other.exec_ctx;
    num_io = other.num_io;
    return (*this);
}

} // namespace execution
} // namespace core
} // namespace trtorch

