#include "core/util/trt_util.h"
#include "core/util/logging/TorchTRTLogger.h"
#include "core/util/macros.h"

namespace torch_tensorrt {
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

  TORCHTRT_CHECK(
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
  TORCHTRT_CHECK(
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
  TORCHTRT_CHECK(
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

  TORCHTRT_CHECK(
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

nvinfer1::Dims unsqueezeDims(const nvinfer1::Dims& d, int pos, int val, bool use_zeros) {
  // acceptable range for pos is [0, d.nbDims]
  TORCHTRT_ASSERT(pos >= 0 && pos <= d.nbDims, "ERROR: Index to unsqueeze is out of bounds.");

  nvinfer1::Dims dims;
  for (int i = 0, j = 0; j <= d.nbDims; j++) {
    // add new dimension at pos
    if (j == pos) {
      dims.d[j] = val;
    } else {
      dims.d[j] = (use_zeros && d.d[i] == -1) ? 0 : d.d[i];
      ++i;
    }
  }

  dims.nbDims = d.nbDims + 1;

  return dims;
}

nvinfer1::Dims squeezeDims(const nvinfer1::Dims& d, int pos, bool use_zeros) {
  // acceptable range for pos is [0, d.nbDims]
  TORCHTRT_ASSERT(pos >= 0 && pos <= d.nbDims, "ERROR: Index to squeeze is out of bounds.");

  nvinfer1::Dims dims;
  int j = 0;
  for (int i = 0; i < d.nbDims; i++) {
    if (i != pos) {
      dims.d[j++] = (use_zeros && d.d[i] == -1) ? 0 : d.d[i];
    }
  }
  dims.nbDims = j;

  return dims;
}

std::vector<int64_t> toVec(nvinfer1::Dims d) {
  std::vector<int64_t> dims;
  for (int i = 0; i < d.nbDims; i++) {
    dims.push_back(d.d[i]);
  }
  return dims;
}

std::string toStr(nvinfer1::Dims d) {
  std::stringstream ss;
  ss << d;
  return ss.str();
}

nvinfer1::DimsHW toDimsHW(c10::List<int64_t> l) {
  TORCHTRT_CHECK(l.size() == 2, "The list requested to be converted to nvinfer1::DimsHW is not 2");

  nvinfer1::DimsHW dims;
  dims.nbDims = l.size();
  for (size_t i = 0; i < l.size(); i++) {
    dims.d[i] = l[i];
  }
  return dims;
}

nvinfer1::DimsHW toDimsHW(c10::IntArrayRef l) {
  TORCHTRT_CHECK(l.size() == 2, "The list requested to be converted to nvinfer1::DimsHW is not 2");

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
      {at::kBool, nvinfer1::DataType::kBOOL}};
  return at_trt_type_map;
}

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_at_type_map() {
  static const std::unordered_map<nvinfer1::DataType, at::ScalarType> trt_at_type_map = {
      {nvinfer1::DataType::kFLOAT, at::kFloat},
      {nvinfer1::DataType::kHALF, at::kHalf},
      {nvinfer1::DataType::kINT32, at::kInt},
      {nvinfer1::DataType::kINT8, at::kChar},
      {nvinfer1::DataType::kBOOL, at::kBool},
  };
  return trt_at_type_map;
}
} // namespace

const std::unordered_map<nvinfer1::DataType, at::ScalarType>& get_trt_aten_type_map() {
  return get_trt_at_type_map();
}

at::ScalarType TRTDataTypeToScalarType(nvinfer1::DataType t) {
  auto type = optTRTDataTypeToScalarType(t);
  TORCHTRT_CHECK(type, "Unsupported TensorRT data type " << t);
  return type.value();
}

c10::optional<at::ScalarType> optTRTDataTypeToScalarType(nvinfer1::DataType t) {
  auto trt_aten_type_map = get_trt_aten_type_map();
  if (trt_aten_type_map.find(t) != trt_aten_type_map.end()) {
    return trt_aten_type_map.at(t);
  } else {
    return {};
  }
}

const std::unordered_map<at::ScalarType, nvinfer1::DataType>& get_aten_trt_type_map() {
  return get_at_trt_type_map();
}

nvinfer1::DataType ScalarTypeToTRTDataType(at::ScalarType t) {
  auto type = optScalarTypeToTRTDataType(t);
  TORCHTRT_CHECK(type, "Unsupported ATen data type " << t);
  return type.value();
}

c10::optional<nvinfer1::DataType> optScalarTypeToTRTDataType(at::ScalarType t) {
  auto aten_trt_type_map = get_aten_trt_type_map();
  if (aten_trt_type_map.find(t) != aten_trt_type_map.end()) {
    return aten_trt_type_map.at(t);
  } else {
    return {};
  }
}

c10::optional<nvinfer1::DataType> optTypeMetaToTRTDataType(caffe2::TypeMeta dtype) {
  if (auto t = c10::optTypeMetaToScalarType(dtype)) {
    return optScalarTypeToTRTDataType(t.value());
  } else {
    return {};
  }
}

torch::jit::Value* getOrAddInputForValue(
    torch::jit::Value* old_value,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  if (old_to_new.count(old_value) == 0) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = graph->createClone(node, {nullptr});
      graph->block()->prependNode(new_const);
      return new_const->output();
    }
    auto new_value = graph->block()->addInput();
    old_to_new[old_value] = new_value;
    new_value->copyMetadata(old_value);
    return new_value;
  } else {
    return old_to_new[old_value];
  }
}

torch::jit::Node* cloneNode(
    torch::jit::Node* node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  auto* block = graph->block();
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v, graph, old_to_new); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(graph->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new[oo] = no;
  }
  return new_node;
}

} // namespace util
} // namespace core
} // namespace torch_tensorrt
