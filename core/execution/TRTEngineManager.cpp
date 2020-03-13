#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {
namespace {
class TRTEngineManager {
public:
    TRTEngineManager()
        : logger_("[TRTorch Execution Manager] - ",
                 util::logging::get_logger().get_reportable_severity(),
                 util::logging::get_logger().get_is_colored_output_on()) {
    }

    TRTEngine* get_engine(EngineID uid) {
        auto iter = engine_registry_.find(uid);

        TRTORCH_ASSERT(iter != engine_registry_.end(), "Unabled to find requested engine (ID: " << uid << ") in TensorRT Execution Manager"); 

        return &(iter->second);
    }
    
    // TODO: Should we have standing engines ready to run or should we be creating execution contexts JIT?
    EngineID register_engine(std::string& serialized_engine) {
        auto engine = TRTEngine(logger_, serialized_engine);
        EngineID uid = engine.id;
        engine_registry_[uid] = std::move(engine);
        LOG_DEBUG(logger_, "Registering new engine (ID: " << std::hex << uid << ") in TensorRT Execution Manager");
        RegisterEngineOp(engine);
        return uid;
    }

    void deregister_engine(EngineID uid) {
        auto iter = engine_registry_.find(uid);
        if (iter == engine_registry_.end()) {
             //TODO: Exception
            LOG_ERROR(logger_, "Unabled to find requested engine (ID: " << uid << ") in TensorRT Execution Manager"); 
        } else {
            auto engine = iter->second;
            // Doing this here since for some reason the destructor causes segfaults
            engine.exec_ctx->destroy();
            engine.cuda_engine->destroy();
            engine_registry_.erase(uid);
        }
    }

private:
    util::logging::TRTorchLogger logger_;
    std::unordered_map<EngineID, TRTEngine> engine_registry_;
};

TRTEngineManager& get_engine_manager() {
    static TRTEngineManager engine_man;
    return engine_man;
}
} // namespace

uint64_t RegisterEngineFromSerializedEngine(std::string& serialized_engine) {
    return get_engine_manager().register_engine(serialized_engine);
}

nvinfer1::ICudaEngine* GetCudaEngine(EngineID id) {
    // Assuming exception will be thrown inside the manager if there is no corresponding engine 
    return get_engine_manager().get_engine(id)->cuda_engine;
}

nvinfer1::IExecutionContext* GetExecCtx(EngineID id) {
     // Assuming exception will be thrown inside the manager if there is no corresponding engine 
     return get_engine_manager().get_engine(id)->exec_ctx;
}

std::pair<uint64_t, uint64_t> GetEngineIO(EngineID id) {
    // Assuming exception will be thrown inside the manager if there is no corresponding engine 
    return get_engine_manager().get_engine(id)->num_io;
}

c10::FunctionSchema GetEngineFunctionSchema(EngineID id) {
    // Assuming exception will be thrown inside the manager if there is no corresponding engine 
    return get_engine_manager().get_engine(id)->schema;
}

void DeregisterEngine(EngineID id) {
    get_engine_manager().deregister_engine(id);
}

} // namespace execution
} // namespace core
} // namespace trtorch


