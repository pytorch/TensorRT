#include <iostream>
#include <sstream>
#include <utility>

#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {
namespace conversion {

std::ostream& operator<<(std::ostream& os, const BuilderSettings& s) {
    os << "Settings requested for TensorRT engine:"                                \
       << "\n    Operating Precision: " << s.op_precision                          \
       << "\n    Make Refittable Engine: " << s.refit                              \
       << "\n    Debuggable Engine: " << s.debug                                   \
       << "\n    Strict Type: " << s.strict_type                                   \
       << "\n    Allow GPU Fallback (if running on DLA): " << s.allow_gpu_fallback \
       << "\n    Min Timing Iterations: " << s.num_min_timing_iters                \
       << "\n    Avg Timing Iterations: " << s.num_avg_timing_iters                \
       << "\n    Max Workspace Size: " << s.workspace_size                         \
       << "\n    Device Type: " << s.device                                        \
       << "\n    Engine Capability: " << s.capability;
    return os;
}

ConversionCtx::ConversionCtx(BuilderSettings build_settings)
    : settings(build_settings), logger("[TRTorch Conversion Context] - ",
                                 util::logging::get_logger().get_reportable_severity(),
                                 util::logging::get_logger().get_is_colored_output_on()) {
    // TODO: Support FP16 and FP32 from JIT information
    builder = nvinfer1::createInferBuilder(logger);
    net = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    LOG_DEBUG(build_settings);
    cfg = builder->createBuilderConfig();

    switch(settings.op_precision) {
    case nvinfer1::DataType::kHALF:
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        input_type = nvinfer1::DataType::kHALF;
        break;
    // case nvinfer1::DataType::kINT8:
    //     cfg->setFlag(nvinfer1::BuilderFlag::kINT8);
    //         input_type = nvinfer1::DataType::kFLOAT;
    //     break;
    case nvinfer1::DataType::kFLOAT:
    default:
        input_type = nvinfer1::DataType::kFLOAT;
        break;
    }

    if (settings.refit) {
        cfg->setFlag(nvinfer1::BuilderFlag::kREFIT);
    }

    if (settings.debug) {
        cfg->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    }

    if (settings.strict_type) {
        cfg->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }

    if (settings.allow_gpu_fallback) {
        cfg->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    cfg->setMinTimingIterations(settings.num_min_timing_iters);
    cfg->setAvgTimingIterations(settings.num_avg_timing_iters);
    cfg->setMaxWorkspaceSize(settings.workspace_size);
    cfg->setDefaultDeviceType(settings.device);
    cfg->setEngineCapability(settings.capability);
}

ConversionCtx::~ConversionCtx() {
    builder->destroy();
    net->destroy();
    cfg->destroy();
    for (auto ptr : builder_resources) {
        free(ptr);
    }
}

nvinfer1::ITensor* ConversionCtx::AssociateValueAndTensor(const torch::jit::Value* value, nvinfer1::ITensor* tensor) {
    tensor->setName(value->debugName().c_str());
    this->value_tensor_map[value] = tensor;
    return tensor;
}

std::string ConversionCtx::SerializeEngine() {
    auto engine = builder->buildEngineWithConfig(*net, *cfg);
    auto serialized_engine = engine->serialize();
    return std::string((const char*)serialized_engine->data(), serialized_engine->size());
}

bool ConversionCtx::CheckLayerAddition(const torch::jit::Node* n) {
    for (auto out : n->outputs()) {
        auto iter = this->value_tensor_map.find(out);
        if (iter == this->value_tensor_map.end()) {
            LOG_WARNING("Node " << util::node_info(n) << " output: " << out->debugName() << " does not have a coresponding output, may potentially indicate a defective converter");
            return false;
        }
    }
    return true;
}

} // namespace conversion
} // namespace core
} // namespace trtorch
