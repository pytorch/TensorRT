#include "torch/csrc/jit/script/module.h"

#include "core/util/prelude.h"
#include "core/compiler.h"

#include "trtorch/trtorch.h"

namespace trtorch {
ExtraInfo::DataType::DataType(c10::ScalarType t) {
    assert(t == at::kHalf || t == at::kFloat /*|| t == at::kChar*/);
    switch (t) {
    case at::kHalf:
        value = DataType::kHalf;
        break;
    case at::kFloat:
    default:
        value = DataType::kFloat;
        break;
    // case at::kChar:
    //     value = DataType::kChar;
    }
}

ExtraInfo::DeviceType::DeviceType(c10::DeviceType t) {
    assert(t == at::kCUDA);
    value = DeviceType::kGPU;
}
    
ExtraInfo::InputRange::InputRange(std::vector<int64_t> opt) {
    this->opt = opt;
    this->min = opt;
    this->max = opt;
}

ExtraInfo::InputRange::InputRange(c10::IntArrayRef opt) {
    this->opt = core::util::toVec(opt);
    this->min = core::util::toVec(opt);
    this->max = core::util::toVec(opt);
}

ExtraInfo::InputRange::InputRange(std::vector<int64_t> min, std::vector<int64_t> opt, std::vector<int64_t> max) {
    this->opt = opt;
    this->min = min;
    this->max = max;
}

ExtraInfo::InputRange::InputRange(c10::IntArrayRef min, c10::IntArrayRef opt, c10::IntArrayRef max) {
    this->opt = core::util::toVec(opt);
    this->min = core::util::toVec(min);
    this->max = core::util::toVec(max);
}

ExtraInfo::ExtraInfo(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
    for (auto in : fixed_sizes) {
        input_ranges.push_back(InputRange(in));
    }
}

ExtraInfo::ExtraInfo(std::vector<std::vector<int64_t>> fixed_sizes) {
    for (auto in : fixed_sizes) {
        input_ranges.push_back(InputRange(in));
    }
}

core::conversion::InputRange to_internal_input_range(ExtraInfo::InputRange i) {
    return core::conversion::InputRange(i.min, i.opt, i.max);
}

std::vector<core::conversion::InputRange> to_vec_internal_input_ranges(std::vector<ExtraInfo::InputRange> external) {
    std::vector<core::conversion::InputRange> internal;
    for (auto range : external) {
        internal.push_back(to_internal_input_range(range));
    }
    return internal;
}

core::conversion::ExtraInfo to_internal_extra_info(ExtraInfo external) {
    core::conversion::ExtraInfo internal(to_vec_internal_input_ranges(external.input_ranges));

    switch(external.op_precision) {
    // case ExtraInfo::DataType::kChar:
    //    internal.engine_settings.op_precision = nvinfer1::DataType::kINT8;
    //    break;
    case ExtraInfo::DataType::kHalf:
        internal.engine_settings.op_precision = nvinfer1::DataType::kHALF;
        break;
    case ExtraInfo::DataType::kFloat:
    default:
        internal.engine_settings.op_precision = nvinfer1::DataType::kFLOAT;
    }
    
    internal.engine_settings.refit = external.refit;     
    internal.engine_settings.debug = external.debug; 
    internal.engine_settings.strict_type = external.strict_type; 
    internal.engine_settings.allow_gpu_fallback = external.allow_gpu_fallback; 

    switch(external.device) {
    case ExtraInfo::DeviceType::kDLA:
        internal.engine_settings.device = nvinfer1::DeviceType::kDLA;
        break;
    case ExtraInfo::DeviceType::kGPU:
    default:
        internal.engine_settings.device = nvinfer1::DeviceType::kGPU;
    }

    switch(external.capability) {
    case ExtraInfo::EngineCapability::kSAFE_GPU:
        internal.engine_settings.capability = nvinfer1::EngineCapability::kSAFE_GPU;
        break;
    case ExtraInfo::EngineCapability::kSAFE_DLA:
        internal.engine_settings.capability = nvinfer1::EngineCapability::kSAFE_DLA;
        break;
    case ExtraInfo::EngineCapability::kDEFAULT:
    default:
        internal.engine_settings.capability = nvinfer1::EngineCapability::kDEFAULT;
            
    }
    
    internal.engine_settings.num_min_timing_iters = external.num_min_timing_iters; 
    internal.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;     
    internal.engine_settings.workspace_size = external.workspace_size;

    return internal;
}

} // namespace trtorch
