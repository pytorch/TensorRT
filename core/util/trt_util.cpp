#include "core/util/trt_util.h"
#include "core/util/logging/TRTorchLogger.h"
#include "core/util/macros.h"

namespace trtorch {
namespace core {
namespace util {

bool broadcastable(nvinfer1::Dims a, nvinfer1::Dims b, bool multidirectional) {
  if (a == b) {
    return true;
  }

  if (multidirectional) {
    nvinfer1::Dims a_dims_eq;
    nvinfer1::Dims b_dims_eq;
    if (a.nbDims > b.nbDims) {
      a_dims_eq = a;
      b_dims_eq = toDimsPad(toVec(b), a.nbDims);
    } else if (a.nbDims < b.nbDims) {
      a_dims_eq = toDimsPad(toVec(a), b.nbDims);
      b_dims_eq = b;
    } else {
      a_dims_eq = a;
      b_dims_eq = b;
    }

    bool broadcastable = true;
    for (int i = 0; i < a_dims_eq.nbDims; i++) {
      if (b_dims_eq.d[i] == a_dims_eq.d[i] || (b_dims_eq.d[i] == 1 || a_dims_eq.d[i] == 1)) {
        continue;
      } else {
        broadcastable = false;
        break;
      }
    }
    return broadcastable;
  } else {
    nvinfer1::Dims b_dims_eq;
    if (a.nbDims > b.nbDims) {
      b_dims_eq = toDimsPad(toVec(b), a.nbDims);
    } else if (a.nbDims < b.nbDims) {
      return false;
    } else {
      b_dims_eq = b;
    }

    bool broadcastable = true;
    for (int i = 0; i < a.nbDims; i++) {
      if (b_dims_eq.d[i] == a.d[i] || b_dims_eq.d[i] == 1) {
        continue;
      } else {
        broadcastable = false;
        break;
      }
    }
    return broadcastable;
  }
}

int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

nvinfer1::Dims toDimsPad(c10::IntArrayRef l, uint64_t pad_to) {
  if (l.size() > pad_to) {
    LOG_DEBUG(
        "Requested padding of dimensions to " << pad_to << " but found " << l.size()
                                              << " dimensions, not going to pad");
    return toDims(l);
  }

  TRTORCH_CHECK(
      pad_to <= nvinfer1::Dims::MAX_DIMS,
      "The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");

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
  TRTORCH_CHECK(
      l.size() <= nvinfer1::Dims::MAX_DIMS,
      "The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");

  nvinfer1::Dims dims;
  dims.nbDims = l.size();
  for (size_t i = 0; i < l.size(); i++) {
    dims.d[i] = l[i];
  }
  return dims;
}

nvinfer1::Dims toDims(c10::List<int64_t> l) {
  TRTORCH_CHECK(
      l.size() <= nvinfer1::Dims::MAX_DIMS,
      "The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");

  nvinfer1::Dims dims;
  dims.nbDims = l.size();
  for (size_t i = 0; i < l.size(); i++) {
    dims.d[i] = l[i];
  }
  return dims;
}

nvinfer1::Dims toDimsPad(c10::List<int64_t> l, uint64_t pad_to) {
  if (l.size() > pad_to) {
    LOG_DEBUG(
        "Requested padding of dimensions to " << pad_to << " but found " << l.size()
                                              << " dimensions, not going to pad");
    return toDims(l);
  }

  TRTORCH_CHECK(
      pad_to <= nvinfer1::Dims::MAX_DIMS,
      "The list requested to be converted to nvinfer1::Dims exceeds the max number of dimensions for TensorRT");

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

nvinfer1::Dims unpadDims(const nvinfer1::Dims& d) {
  nvinfer1::Dims dims;

  int j = 0;
  bool pad_dims_done = false;

  for (int i = 0; i < d.nbDims; i++) {
    if (d.d[i] == 1 && !pad_dims_done) {
      // skip over unecessary dimension
      continue;
    } else {
      dims.d[j] = d.d[i];
      j++;

      // keep all other dimensions (don't skip over them)
      pad_dims_done = true;
    }
  }

  dims.nbDims = j;

  return dims;
}

nvinfer1::Dims unsqueezeDims(const nvinfer1::Dims& d, int pos) {
  // acceptable range for pos is [0, d.nbDims]
  TRTORCH_ASSERT(pos >= 0 && pos <= d.nbDims, "ERROR: Index to unsqueeze is out of bounds.");

  nvinfer1::Dims dims;

  int i = 0;
  int j = 0;

  while (i <= d.nbDims) {
    if (j != pos) {
      dims.d[j] = d.d[i];
      i++;
    } else {
      // add new dimension at pos
      dims.d[j] = 1;
    }

    j++;
  }

  dims.nbDims = d.nbDims + 1;

  return dims;
}

nvinfer1::Dims squeezeDims(const nvinfer1::Dims& d, int pos) {
  // acceptable range for pos is [0, d.nbDims]
  TRTORCH_ASSERT(pos >= 0 && pos <= d.nbDims, "ERROR: Index to squeeze is out of bounds.");

  nvinfer1::Dims dims;

  int i = 0;
  int j = 0;

  while (i <= d.nbDims) {
    if (j != pos) {
      dims.d[j] = d.d[i];
    } else {
      // add new dimension at pos
      i++;
      if (i <= d.nbDims) {
        dims.d[j] = d.d[i];
      }
      
    }
    i++;
    j++;
    
  }

  dims.nbDims = d.nbDims - 1;

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
  TRTORCH_CHECK(l.size() == 2, "The list requested to be converted to nvinfer1::DimsHW is not 2");

  nvinfer1::DimsHW dims;
  dims.nbDims = l.size();
  for (size_t i = 0; i < l.size(); i++) {
    dims.d[i] = l[i];
  }
  return dims;
}

nvinfer1::DimsHW toDimsHW(c10::IntArrayRef l) {
  TRTORCH_CHECK(l.size() == 2, "The list requested to be converted to nvinfer1::DimsHW is not 2");

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

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_at_type_map() {
  static const std::unordered_map<nvinfer1::DataType, at::ScalarType> trt_at_type_map = {
      {nvinfer1::DataType::kFLOAT, at::kFloat},
      {nvinfer1::DataType::kHALF, at::kHalf},
      {nvinfer1::DataType::kINT32, at::kInt},
      {nvinfer1::DataType::kINT8, at::kChar},
  };
  return trt_at_type_map;
}
} // namespace

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_aten_type_map() {
  return get_trt_at_type_map();
}

at::ScalarType toATenDType(nvinfer1::DataType t) {
  return get_trt_aten_type_map().at(t);
}

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
