#pragma once

#include "ATen/Tensor.h"
#include "ATen/core/List.h"
#include "NvInfer.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/ir/ir.h"

namespace nvinfer1 {

#if NV_TENSORRT_MAJOR < 8

#define TRT_ENGINE_CAPABILITY_STANDARD nvinfer1::EngineCapability::kDEFAULT
#define TRT_ENGINE_CAPABILITY_SAFETY nvinfer1::EngineCapability::kSAFE_GPU
#define TRT_ENGINE_CAPABILITY_DLA_STANDALONE nvinfer1::EngineCapability::kSAFE_DLA

template <class T>
std::shared_ptr<T> make_trt(T* p) {
  return std::shared_ptr<T>(p, [](T* p) { p->destroy(); });
}

#else

#define TRT_ENGINE_CAPABILITY_STANDARD nvinfer1::EngineCapability::kSTANDARD
#define TRT_ENGINE_CAPABILITY_SAFETY nvinfer1::EngineCapability::kSAFETY
#define TRT_ENGINE_CAPABILITY_DLA_STANDALONE nvinfer1::EngineCapability::kDLA_STANDALONE

template <class T>
std::shared_ptr<T> make_trt(T* p) {
  return std::shared_ptr<T>(p);
}

#endif

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::TensorFormat& format) {
  switch (format) {
    case nvinfer1::TensorFormat::kLINEAR:
      return os << "NCHW\\Contiguous\\Linear";
    case nvinfer1::TensorFormat::kHWC:
      return os << "NHWC\\Channel Last";
    default:
      return os << "Unknown Tensor Format";
  }
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::DataType& dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return stream << "Float32";
    case nvinfer1::DataType::kHALF:
      return stream << "Float16";
    case nvinfer1::DataType::kINT8:
      return stream << "Int8";
    case nvinfer1::DataType::kINT32:
      return stream << "Int32";
    case nvinfer1::DataType::kBOOL:
      return stream << "Bool";
    default:
      return stream << "Unknown Data Type";
  }
}

inline bool operator==(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
  if (in1.nbDims != in2.nbDims) {
    return false;
  }

  // TODO maybe look to support broadcasting comparisons

  for (int64_t i = 0; i < in1.nbDims; i++) {
    if (in1.d[i] != in2.d[i]) {
      return false;
    }
  }
  return true;
}

inline bool operator!=(const nvinfer1::Dims& in1, const nvinfer1::Dims& in2) {
  return !(in1 == in2);
}

template <typename T>
inline std::ostream& printSequence(std::ostream& stream, const T* begin, int count) {
  stream << "[";
  if (count > 0) {
    std::copy_n(begin, count - 1, std::ostream_iterator<T>(stream, ", "));
    stream << begin[count - 1];
  }
  stream << "]";
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::Dims& shape) {
  return printSequence(stream, shape.d, shape.nbDims);
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::Permutation& perm) {
  return printSequence(stream, perm.order, nvinfer1::Dims::MAX_DIMS);
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::DeviceType& dtype) {
  switch (dtype) {
    case nvinfer1::DeviceType::kGPU:
      return stream << "GPU";
    case nvinfer1::DeviceType::kDLA:
      return stream << "DLA";
    default:
      return stream << "Unknown Device Type";
  }
}

inline std::ostream& operator<<(std::ostream& stream, const nvinfer1::EngineCapability& cap) {
  switch (cap) {
    case TRT_ENGINE_CAPABILITY_STANDARD:
      return stream << "standard";
    case TRT_ENGINE_CAPABILITY_SAFETY:
      return stream << "safety";
    case TRT_ENGINE_CAPABILITY_DLA_STANDALONE:
      return stream << "DLA standalone";
    default:
      return stream << "Unknown Engine Capability Setting";
  }
}
} // namespace nvinfer1

namespace torch_tensorrt {
namespace core {
namespace util {

int64_t volume(const nvinfer1::Dims& d);

bool broadcastable(nvinfer1::Dims a, nvinfer1::Dims b, bool multidirectional = true);
nvinfer1::Dims toDimsPad(c10::IntArrayRef l, uint64_t pad_to);
nvinfer1::Dims toDimsPad(c10::List<int64_t> l, uint64_t pad_to);
nvinfer1::Dims toDimsTailPad(c10::IntArrayRef l, uint64_t pad_to);
nvinfer1::Dims toDimsTailPad(c10::List<int64_t> l, uint64_t pad_to);
nvinfer1::Dims unpadDims(const nvinfer1::Dims& d);
nvinfer1::Dims unsqueezeDims(const nvinfer1::Dims& d, int pos, int val = 1, bool use_zeros = true);
nvinfer1::Dims squeezeDims(const nvinfer1::Dims& d, int pos, bool use_zeros = true);
nvinfer1::Dims toDims(c10::IntArrayRef l);
nvinfer1::Dims toDims(c10::List<int64_t> l);
nvinfer1::DimsHW toDimsHW(c10::List<int64_t> l);
nvinfer1::DimsHW toDimsHW(c10::IntArrayRef l);
std::vector<int64_t> toVec(nvinfer1::Dims d);
std::string toStr(nvinfer1::Dims d);

at::ScalarType TRTDataTypeToScalarType(nvinfer1::DataType t);
c10::optional<at::ScalarType> optTRTDataTypeToScalarType(nvinfer1::DataType t);
nvinfer1::DataType ScalarTypeToTRTDataType(at::ScalarType t);
c10::optional<nvinfer1::DataType> optScalarTypeToTRTDataType(at::ScalarType t);
c10::optional<nvinfer1::DataType> optTypeMetaToTRTDataType(caffe2::TypeMeta dtype);
torch::jit::Value* getOrAddInputForValue(
    torch::jit::Value* old_value,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new);
torch::jit::Node* cloneNode(
    torch::jit::Node* node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new);
const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_aten_trt_type_map();

} // namespace util
} // namespace core
} // namespace torch_tensorrt
