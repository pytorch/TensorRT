#include <algorithm>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string slugify(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  return s;
}

std::vector<std::string> split(const std::string& str, char delim) {
  std::vector<std::string> strings;
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    strings.push_back(str.substr(start, end - start));
  }
  return strings;
}

TRTEngine::TRTEngine(
    std::string serialized_engine,
    CUDADevice cuda_device,
    const std::vector<std::string>& _in_binding_names,
    const std::vector<std::string>& _out_binding_names) {
  std::string _name = "deserialized_trt";
  new (this) TRTEngine(_name, serialized_engine, cuda_device, _in_binding_names, _out_binding_names);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info) {
  TORCHTRT_CHECK(
      serialized_info.size() == SERIALIZATION_LEN,
      "Program to be deserialized targets an incompatible Torch-TensorRT ABI");
  TORCHTRT_CHECK(
      serialized_info[ABI_TARGET_IDX] == ABI_VERSION,
      "Program to be deserialized targets a different Torch-TensorRT ABI Version ("
          << serialized_info[ABI_TARGET_IDX] << ") than the Torch-TensorRT Runtime ABI Version (" << ABI_VERSION
          << ")");
  std::string _name = serialized_info[NAME_IDX];
  std::string engine_info = serialized_info[ENGINE_IDX];
  std::vector<std::string> in_bindings = split(serialized_info[INPUT_BINDING_NAMES_IDX], '%');
  std::vector<std::string> out_bindings = split(serialized_info[OUTPUT_BINDING_NAMES_IDX], '%');

  CUDADevice cuda_device(serialized_info[DEVICE_IDX]);

  new (this) TRTEngine(_name, engine_info, cuda_device, in_bindings, out_bindings);
}

TRTEngine::TRTEngine(
    std::string mod_name,
    std::string serialized_engine,
    CUDADevice cuda_device,
    const std::vector<std::string>& _in_binding_names,
    const std::vector<std::string>& _out_binding_names) {
  auto most_compatible_device = get_most_compatible_device(cuda_device);
  TORCHTRT_CHECK(most_compatible_device, "No compatible device was found for instantiating TensorRT engine");
  device_info = most_compatible_device.value();
  set_cuda_device(device_info);

  rt = make_trt(nvinfer1::createInferRuntime(util::logging::get_logger()));

  name = slugify(mod_name);

  cuda_engine = make_trt(rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size()));
  TORCHTRT_CHECK((cuda_engine.get() != nullptr), "Unable to deserialize the TensorRT engine");

  exec_ctx = make_trt(cuda_engine->createExecutionContext());
  TORCHTRT_CHECK((exec_ctx.get() != nullptr), "Unable to create TensorRT execution context");

  if (_in_binding_names.size() == 0 && _out_binding_names.size() == 0) {
    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
      std::string bind_name = cuda_engine->getBindingName(x);
      LOG_DEBUG("Binding name: " << bind_name);
      auto delim = bind_name.find(".");
      if (delim == std::string::npos) {
        delim = bind_name.find("_");
        TORCHTRT_CHECK(
            delim != std::string::npos,
            "Unable to determine binding index for input "
                << bind_name
                << "\nEnsure module was compiled with Torch-TensorRT.ts or follows Torch-TensorRT Runtime conventions");
      }

      std::string idx_s = bind_name.substr(delim + 1);
      uint64_t idx = static_cast<uint64_t>(std::stoi(idx_s));

      if (cuda_engine->bindingIsInput(x)) {
        inputs++;
        in_binding_map[x] = idx;
        LOG_DEBUG("TRT Binding: " << x << ": PYT Input: " << idx);
      } else {
        outputs++;
        out_binding_map[x] = idx;
        LOG_DEBUG("TRT Binding: " << x << ": PYT Output: " << idx);
      }
    }

    num_io = std::make_pair(inputs, outputs);
    in_binding_names.resize(inputs);
    out_binding_names.resize(outputs);

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
      std::cout << x << std::endl;
      std::string bind_name = cuda_engine->getBindingName(x);
      if (cuda_engine->bindingIsInput(x)) {
        in_binding_names[in_binding_map.at(x)] = bind_name;
      } else {
        out_binding_names[out_binding_map.at(x)] = bind_name;
      }
    }
  } else {
    uint64_t inputs = _in_binding_names.size();
    in_binding_names.resize(inputs);
    for (size_t pyt_idx = 0; pyt_idx < inputs; pyt_idx++) {
      auto binding_name = _in_binding_names[pyt_idx];
      auto trt_idx = cuda_engine->getBindingIndex(binding_name.c_str());
      TORCHTRT_CHECK((trt_idx >= 0), "Could not find a TensorRT engine binding for input named " << binding_name);
      TORCHTRT_CHECK(
          cuda_engine->bindingIsInput(trt_idx),
          "Binding " << binding_name << " specified as input but found as output in TensorRT engine");
      LOG_DEBUG(
          "Input binding name: " << binding_name << " (trt binding idx: " << trt_idx << ", "
                                 << "pyt arg idx: " << pyt_idx << ")");
      in_binding_map[trt_idx] = pyt_idx;
      in_binding_names[pyt_idx] = _in_binding_names[pyt_idx];
    }

    uint64_t outputs = _out_binding_names.size();
    out_binding_names.resize(outputs);
    for (size_t pyt_idx = 0; pyt_idx < outputs; pyt_idx++) {
      auto binding_name = _out_binding_names[pyt_idx];
      auto trt_idx = cuda_engine->getBindingIndex(binding_name.c_str());
      TORCHTRT_CHECK((trt_idx >= 0), "Could not find a TensorRT engine binding for output named " << binding_name);
      TORCHTRT_CHECK(
          !cuda_engine->bindingIsInput(trt_idx),
          "Binding " << binding_name << " specified as output but found as input in TensorRT engine");
      LOG_DEBUG(
          "Output binding name: " << binding_name << " (trt binding idx: " << trt_idx << ", "
                                  << "pyt return idx: " << pyt_idx << ")");
      out_binding_map[trt_idx] = pyt_idx;
      out_binding_names[pyt_idx] = binding_name;
    }
    num_io = std::make_pair(inputs, outputs);
  }

  LOG_DEBUG(*this);
}

void TRTEngine::set_paths() {
  execution_profile_path = profile_path + "/" + name + "_execution_profile.trace";
  device_profile_path = profile_path + "/" + name + "_device_config_profile.trace";
  input_profile_path = profile_path + "/" + name + "_input_profile.trace";
  output_profile_path = profile_path + "/" + name + "_output_profile.trace";
  enqueue_profile_path = profile_path + "/" + name + "_enqueue_profile.trace";
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
  rt = other.rt;
  cuda_engine = other.cuda_engine;
  device_info = other.device_info;
  exec_ctx = other.exec_ctx;
  num_io = other.num_io;
  return (*this);
}

std::string TRTEngine::to_str() const {
  // clang-format off
  std::stringstream ss;
  ss << "Torch-TensorRT TensorRT Engine:" << std::endl;
  ss << "  Name: " << name << std::endl;
  ss << "  Bindings: {" << std::endl;
  for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
    if (cuda_engine->bindingIsInput(x)) {
      const uint64_t pyt_idx = in_binding_map.at(x);
  ss << "    (" << x << ": " << in_binding_names.at(pyt_idx) << ") Input: [" << std::endl;
  ss << "      pytorch arg idx: " << pyt_idx << std::endl;
  ss << "        shape: " << exec_ctx->getBindingDimensions(x) << std::endl;
  ss << "        dtype: " << util::TRTDataTypeToScalarType(exec_ctx->getEngine().getBindingDataType(x)) << std::endl;
  ss << "    ]" << std::endl;
    } else {
      const uint64_t pyt_idx = out_binding_map.at(x);
  ss << "    (" << x <<  ": " << out_binding_names.at(pyt_idx) << ") Output: [" << std::endl;
  ss << "      pytorch return idx: " << pyt_idx << std::endl;
  ss << "        shape: " << exec_ctx->getBindingDimensions(x) << std::endl;
  ss << "        dtype: " << util::TRTDataTypeToScalarType(exec_ctx->getEngine().getBindingDataType(x)) << std::endl;
  ss << "    ]" << std::endl;
    }
  }
  ss << "  }" << std::endl;
  ss << "  Device: " << device_info << std::endl;
  // clang-format on
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TRTEngine& engine) {
  os << engine.to_str();
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
