#pragma once
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace trtorch {
namespace core {
namespace runtime {

using EngineID = int64_t;

typedef enum { DEVICE_IDX = 0, ENGINE_IDX } SerializedInfoIndex;

struct CudaDevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  size_t device_name_len;
  std::string device_name;

  int64_t get_id(void) {
    return this->id;
  }

  void set_id(int64_t gpu_id) {
    this->id = gpu_id;
  }

  int64_t get_major(void) {
    return major;
  }

  void set_major(int64_t major_version) {
    this->major = major_version;
  }

  int64_t get_minor(void) {
    return minor;
  }

  void set_minor(int64_t minor_version) {
    this->minor = minor_version;
  }

  nvinfer1::DeviceType get_device_type(void) {
    return device_type;
  }

  void set_device_type(nvinfer1::DeviceType dev_type = nvinfer1::DeviceType::kGPU) {
    this->device_type = dev_type;
  }

  std::string get_device_name(void) {
    return device_name;
  }

  void set_device_name(std::string& name) {
    this->device_name = name;
  }

  size_t get_device_name_len(void) {
    return device_name_len;
  }

  void set_device_name_len(size_t size) {
    this->device_name_len = size;
  }
};

void set_cuda_device(CudaDevice& cuda_device);
void get_cuda_device(CudaDevice& cuda_device);

std::string serialize_device(CudaDevice& cuda_device);
CudaDevice deserialize_device(std::string device_info);

CudaDevice get_device_info(int64_t gpu_id, nvinfer1::DeviceType device_type);

struct TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  nvinfer1::IRuntime* rt;
  nvinfer1::ICudaEngine* cuda_engine;
  nvinfer1::IExecutionContext* exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
  EngineID id;
  std::string name;
  CudaDevice device_info;
  util::logging::TRTorchLogger logger;

  std::unordered_map<uint64_t, uint64_t> in_binding_map;
  std::unordered_map<uint64_t, uint64_t> out_binding_map;

  ~TRTEngine();
  TRTEngine(std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine(std::vector<std::string> serialized_info);
  TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine& operator=(const TRTEngine& other);
  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

class DeviceList {
  using DeviceMap = std::unordered_map<int, CudaDevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList(void) {
    int num_devices = 0;
    auto status = cudaGetDeviceCount(&num_devices);
    TRTORCH_ASSERT((status == cudaSuccess), "Unable to read CUDA capable devices. Return status: " << status);
    cudaDeviceProp device_prop;
    for (int i = 0; i < num_devices; i++) {
      TRTORCH_CHECK(
          (cudaGetDeviceProperties(&device_prop, i) == cudaSuccess),
          "Unable to read CUDA Device Properies for device id: " << i);
      std::string device_name(device_prop.name);
      CudaDevice device = {
          i, device_prop.major, device_prop.minor, nvinfer1::DeviceType::kGPU, device_name.size(), device_name};
      device_list[i] = device;
    }
  }

 public:
  static DeviceList& instance() {
    static DeviceList obj;
    return obj;
  }

  void insert(int device_id, CudaDevice cuda_device) {
    device_list[device_id] = cuda_device;
  }
  CudaDevice find(int device_id) {
    return device_list[device_id];
  }
  DeviceMap get_devices() {
    return device_list;
  }
};

namespace {
static DeviceList cuda_device_list;
}

} // namespace runtime
} // namespace core
} // namespace trtorch
