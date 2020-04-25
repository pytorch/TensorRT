#include <algorithm>

#include "NvInfer.h"
#include "torch/csrc/jit/script/function_schema_parser.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {

std::string slugify(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

c10::FunctionSchema GenerateEngineFunctionSchema(EngineID id, nvinfer1::ICudaEngine* engine, uint64_t num_inputs, uint64_t num_outputs) {
    std::stringstream ss;
    ss << "trt::execute_engine_" << std::hex << id << "(";

    std::stringstream in_ss;
    std::stringstream out_ss;

    uint64_t inputs_parsed = 0;
    uint64_t outputs_parsed = 0;
    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            in_ss << "Tensor in_";
            in_ss << slugify(engine->getBindingName(i));
            if (inputs_parsed + 1 < num_inputs) {
                in_ss << ", ";
                inputs_parsed++;
            }
        } else {
            out_ss << "Tensor";
            if (outputs_parsed + 1 < num_outputs) {
                out_ss << ", ";
                outputs_parsed++;
            }
        }
    }

    ss << in_ss.str();
    ss << ") -> (";
    ss << out_ss.str();
    ss << ')';
    return torch::jit::parseSchema(ss.str());
}

TRTEngine::TRTEngine()
    : schema(torch::jit::parseSchema("trt::noop() -> ()")) {
}

TRTEngine::TRTEngine(nvinfer1::ILogger& logger, std::string& serialized_engine)
    : schema(torch::jit::parseSchema("trt::noop() -> ()")) { // Need a better default

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
    schema = GenerateEngineFunctionSchema(id, cuda_engine, inputs, outputs);
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
    id = other.id;
    rt = other.rt;
    cuda_engine = other.cuda_engine;
    exec_ctx = other.exec_ctx;
    num_io = other.num_io;
    schema = other.schema;
    return (*this);
}

} // namespace execution
} // namespace core
} // namespace trtorch

