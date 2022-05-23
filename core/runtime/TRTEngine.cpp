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

TRTEngine::TRTEngine(std::string serialized_engine, CudaDevice cuda_device) {
  std::string _name = "deserialized_trt";
  new (this) TRTEngine(_name, serialized_engine, cuda_device);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info) {
  TORCHTRT_CHECK(
      serialized_info.size() == ENGINE_IDX + 1,
      "Program to be deserialized targets an incompatible Torch-TensorRT ABI");
  TORCHTRT_CHECK(
      serialized_info[ABI_TARGET_IDX] == ABI_VERSION,
      "Program to be deserialized targets a different Torch-TensorRT ABI Version ("
          << serialized_info[ABI_TARGET_IDX] << ") than the Torch-TensorRT Runtime ABI Version (" << ABI_VERSION
          << ")");
  std::string _name = serialized_info[NAME_IDX];
  std::string engine_info = serialized_info[ENGINE_IDX];

  CudaDevice cuda_device(serialized_info[DEVICE_IDX]);
  new (this) TRTEngine(_name, engine_info, cuda_device);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device) {
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
    } else {
      outputs++;
      out_binding_map[x] = idx;
    }
  }
  num_io = std::make_pair(inputs, outputs);

  LOG_DEBUG(*this);
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
  std::stringstream ss;
  ss << "Torch-TensorRT TensorRT Engine:" << std::endl;
  ss << "  Name: " << name << std::endl;
  ss << "  Inputs: [" << std::endl;
  for (uint64_t i = 0; i < num_io.first; i++) {
    ss << "    id: " << i << std::endl;
    ss << "      shape: " << exec_ctx->getBindingDimensions(i) << std::endl;
    ss << "      dtype: " << util::TRTDataTypeToScalarType(exec_ctx->getEngine().getBindingDataType(i)) << std::endl;
  }
  ss << "  ]" << std::endl;
  ss << "  Outputs: [" << std::endl;
  for (uint64_t o = 0; o < num_io.second; o++) {
    ss << "    id: " << o << std::endl;
    ss << "      shape: " << exec_ctx->getBindingDimensions(o) << std::endl;
    ss << "      dtype: " << util::TRTDataTypeToScalarType(exec_ctx->getEngine().getBindingDataType(o)) << std::endl;
  }
  ss << "  ]" << std::endl;
  ss << "  Device: " << device_info << std::endl;

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TRTEngine& engine) {
  os << engine.to_str();
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
