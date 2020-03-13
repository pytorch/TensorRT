#include "core/util/macros.h"
#include "core/util/trt_util.h"
#include "core/util/logging/TRTorchLogger.h"

namespace trtorch {
namespace core {
namespace util {

int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

nvinfer1::Dims toDimsPad(c10::IntArrayRef l, uint64_t pad_to) {
    if (l.size() > pad_to) {
        LOG_DEBUG("Requested padding of dimensions to " << pad_to << " but found " << l.size() << " dimensions, not going to pad");
        return toDims(l);
    }
    
    if (pad_to > nvinfer1::Dims::MAX_DIMS) {
        //TODO: Handle this with exceptions or whatever
        LOG_INTERNAL_ERROR("The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");
    }
    
    nvinfer1::Dims dims;
    dims.nbDims = pad_to;
    for (size_t i = 0; i < pad_to - l.size(); i++) {
        dims.d[i] = 1;
    }
    
    for (size_t i = pad_to - l.size(); i < pad_to; i++) {
        dims.d[i] = l[i - (pad_to - l.size())];
    }
    return dims;
}

nvinfer1::Dims toDims(c10::IntArrayRef l) {
    if (l.size() > nvinfer1::Dims::MAX_DIMS) {
        //TODO: Handle this with exceptions or whatever
        LOG_INTERNAL_ERROR("The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");
    }
    nvinfer1::Dims dims;
    dims.nbDims = l.size();
    for (size_t i = 0; i < l.size(); i++) {
        dims.d[i] = l[i];
    }
    return dims;
}

nvinfer1::Dims toDims(c10::List<int64_t> l) {
    if (l.size() > nvinfer1::Dims::MAX_DIMS) {
        //TODO: Handle this with exceptions or whatever
        LOG_INTERNAL_ERROR("The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");
    }
    nvinfer1::Dims dims;
    dims.nbDims = l.size();
    for (size_t i = 0; i < l.size(); i++) {
        dims.d[i] = l[i];
    }
    return dims;
}
    
std::vector<int64_t> toVec(nvinfer1::Dims d) {
    std::vector<int64_t> dims;
    for (int i = 0; i < d.nbDims; i++) {
        dims.push_back(d.d[i]);
    }
    return std::move(dims);
}

std::string toStr(nvinfer1::Dims d) {
    std::stringstream ss;
    ss << d;
    return ss.str();
}


nvinfer1::DimsHW toDimsHW(c10::List<int64_t> l) {
    if (l.size() != 2) {
        //TODO: Handle this with exceptions or whatever
        LOG_INTERNAL_ERROR("The list requested to be converted to nvinfer1::DimsHW is not 2");
    }
    nvinfer1::DimsHW dims;
    dims.nbDims = l.size();
    for (size_t i = 0; i < l.size(); i++) {
        dims.d[i] = l[i];
    }
    return dims;
}

nvinfer1::DimsHW toDimsHW(c10::IntArrayRef l) {
    if (l.size() != 2) {
        //TODO: Handle this with exceptions or whatever
        LOG_INTERNAL_ERROR("The list requested to be converted to nvinfer1::DimsHW is not 2");
    }
    nvinfer1::DimsHW dims;
    dims.nbDims = l.size();
    for (size_t i = 0; i < l.size(); i++) {
        dims.d[i] = l[i];
    }
    return dims;
}

namespace {
const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_at_trt_type_map() {
    static const std::unordered_map<at::ScalarType, nvinfer1::DataType> at_trt_type_map = {
        {at::kFloat, nvinfer1::DataType::kFLOAT},
        {at::kHalf, nvinfer1::DataType::kHALF},
        {at::kInt, nvinfer1::DataType::kINT32},
        {at::kChar, nvinfer1::DataType::kINT8},
    };
    return at_trt_type_map;
}
} // namespace

const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_aten_trt_type_map() {
    return get_at_trt_type_map();
}

nvinfer1::DataType toTRTDataType(at::ScalarType t) {
    return get_aten_trt_type_map().at(t);
}

c10::optional<nvinfer1::DataType> toTRTDataType(caffe2::TypeMeta dtype) {
    if (auto t = c10::tryTypeMetaToScalarType(dtype)) {
        return toTRTDataType(t.value());
    } else {
        return {};
    }
}

} // namespace util
} // namespace core
} // namespace trtorch
