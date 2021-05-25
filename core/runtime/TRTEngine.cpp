#include <algorithm>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace runtime {

std::string slugify(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  return s;
}

TRTEngine::TRTEngine(std::string serialized_engine, CudaDevice cuda_device)
    : logger(
          std::string("[] - "),
          util::logging::get_logger().get_reportable_severity(),
          util::logging::get_logger().get_is_colored_output_on()) {
  std::string _name = "deserialized_trt";
  new (this) TRTEngine(_name, serialized_engine, cuda_device);
}

TRTEngine::TRTEngine(std::vector<std::string> serialized_info)
    : logger(
          std::string("[] = "),
          util::logging::get_logger().get_reportable_severity(),
          util::logging::get_logger().get_is_colored_output_on()) {
  std::string _name = "deserialized_trt";
  std::string engine_info = serialized_info[EngineIdx];

  CudaDevice cuda_device = deserialize_device(serialized_info[DeviceIdx]);
  new (this) TRTEngine(_name, engine_info, cuda_device);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device)
    : logger(
          std::string("[") + mod_name + std::string("_engine] - "),
          util::logging::get_logger().get_reportable_severity(),
          util::logging::get_logger().get_is_colored_output_on()) {
  device_info = cuda_device;
  set_cuda_device(device_info);

  rt = nvinfer1::createInferRuntime(logger);

  name = slugify(mod_name) + "_engine";

  cuda_engine = rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size());
  // Easy way to get a unique name for each engine, maybe there is a more
  // descriptive way (using something associated with the graph maybe)
  id = reinterpret_cast<EngineID>(cuda_engine);

  exec_ctx = cuda_engine->createExecutionContext();

  uint64_t inputs = 0;
  uint64_t outputs = 0;

  for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
    std::string name = cuda_engine->getBindingName(x);
    std::string idx_s = name.substr(name.find("_") + 1);
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
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
  id = other.id;
  rt = other.rt;
  cuda_engine = other.cuda_engine;
  device_info = other.device_info;
  exec_ctx = other.exec_ctx;
  num_io = other.num_io;
  return (*this);
}

TRTEngine::~TRTEngine() {
  exec_ctx->destroy();
  cuda_engine->destroy();
  rt->destroy();
}

// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }

namespace {
static auto TRTORCH_UNUSED TRTEngineTSRegistrtion =
    torch::class_<TRTEngine>("tensorrt", "Engine")
        .def(torch::init<std::vector<std::string>>())
        // TODO: .def("__call__", &TRTEngine::Run)
        // TODO: .def("run", &TRTEngine::Run)
        .def_pickle(
            [](const c10::intrusive_ptr<TRTEngine>& self) -> std::vector<std::string> {
              // Serialize TensorRT engine
              auto serialized_trt_engine = self->cuda_engine->serialize();

              // Adding device info related meta data to the serialized file
              auto trt_engine = std::string((const char*)serialized_trt_engine->data(), serialized_trt_engine->size());

              std::vector<std::string> serialize_info;
              serialize_info.push_back(serialize_device(self->device_info));
              serialize_info.push_back(trt_engine);
              return serialize_info;
            },
            [](std::vector<std::string> seralized_info) -> c10::intrusive_ptr<TRTEngine> {
              return c10::make_intrusive<TRTEngine>(std::move(seralized_info));
            });
} // namespace
void set_cuda_device(CudaDevice& cuda_device) {
  TRTORCH_CHECK((cudaSetDevice(cuda_device.id) == cudaSuccess), "Unable to set device: " << cuda_device.id);
}

void get_cuda_device(CudaDevice& cuda_device) {
  int device = 0;
  TRTORCH_CHECK(
      (cudaGetDevice(reinterpret_cast<int*>(&device)) == cudaSuccess),
      "Unable to get current device: " << cuda_device.id);
  cuda_device.id = static_cast<int64_t>(device);
  cudaDeviceProp device_prop;
  TRTORCH_CHECK(
      (cudaGetDeviceProperties(&device_prop, cuda_device.id) == cudaSuccess),
      "Unable to get CUDA properties from device:" << cuda_device.id);
  cuda_device.set_major(device_prop.major);
  cuda_device.set_minor(device_prop.minor);
  std::string device_name(device_prop.name);
  cuda_device.set_device_name(device_name);
}

std::string serialize_device(CudaDevice& cuda_device) {
  void* buffer = new char[sizeof(cuda_device)];
  void* ref_buf = buffer;

  int64_t temp = cuda_device.get_id();
  memcpy(buffer, reinterpret_cast<int64_t*>(&temp), sizeof(int64_t));
  buffer = static_cast<char*>(buffer) + sizeof(int64_t);

  temp = cuda_device.get_major();
  memcpy(buffer, reinterpret_cast<int64_t*>(&temp), sizeof(int64_t));
  buffer = static_cast<char*>(buffer) + sizeof(int64_t);

  temp = cuda_device.get_minor();
  memcpy(buffer, reinterpret_cast<int64_t*>(&temp), sizeof(int64_t));
  buffer = static_cast<char*>(buffer) + sizeof(int64_t);

  auto device_type = cuda_device.get_device_type();
  memcpy(buffer, reinterpret_cast<char*>(&device_type), sizeof(nvinfer1::DeviceType));
  buffer = static_cast<char*>(buffer) + sizeof(nvinfer1::DeviceType);

  size_t device_name_len = cuda_device.get_device_name_len();
  memcpy(buffer, reinterpret_cast<char*>(&device_name_len), sizeof(size_t));
  buffer = static_cast<char*>(buffer) + sizeof(size_t);

  auto device_name = cuda_device.get_device_name();
  memcpy(buffer, reinterpret_cast<char*>(&device_name), device_name.size());
  buffer = static_cast<char*>(buffer) + device_name.size();

  return std::string((const char*)ref_buf, sizeof(int64_t) * 3 + sizeof(nvinfer1::DeviceType) + device_name.size());
}

CudaDevice deserialize_device(std::string device_info) {
  CudaDevice ret;
  char* buffer = new char[device_info.size() + 1];
  std::copy(device_info.begin(), device_info.end(), buffer);
  int64_t temp = 0;

  memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int64_t));
  buffer += sizeof(int64_t);
  ret.set_id(temp);

  memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int64_t));
  buffer += sizeof(int64_t);
  ret.set_major(temp);

  memcpy(&temp, reinterpret_cast<char*>(buffer), sizeof(int64_t));
  buffer += sizeof(int64_t);
  ret.set_minor(temp);

  nvinfer1::DeviceType device_type;
  memcpy(&device_type, reinterpret_cast<char*>(buffer), sizeof(nvinfer1::DeviceType));
  buffer += sizeof(nvinfer1::DeviceType);

  size_t size;
  memcpy(&size, reinterpret_cast<size_t*>(&buffer), sizeof(size_t));
  buffer += sizeof(size_t);

  ret.set_device_name_len(size);

  std::string device_name;
  memcpy(&device_name, reinterpret_cast<char*>(buffer), size * sizeof(char));
  buffer += size * sizeof(char);

  ret.set_device_name(device_name);

  return ret;
}

CudaDevice get_device_info(int64_t gpu_id, nvinfer1::DeviceType device_type) {
  CudaDevice cuda_device;
  cudaDeviceProp device_prop;

  // Device ID
  cuda_device.set_id(gpu_id);

  // Get Device Properties
  cudaGetDeviceProperties(&device_prop, gpu_id);

  // Compute capability major version
  cuda_device.set_major(device_prop.major);

  // Compute capability minor version
  cuda_device.set_minor(device_prop.minor);

  std::string device_name(device_prop.name);

  // Set Device name
  cuda_device.set_device_name(device_name);

  // Set Device name len for serialization/deserialization
  cuda_device.set_device_name_len(device_name.size());

  // Set Device Type
  cuda_device.set_device_type(device_type);

  return cuda_device;
}

} // namespace runtime
} // namespace core
} // namespace trtorch
