#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "3";
typedef enum { ABI_TARGET_IDX = 0, NAME_IDX, DEVICE_IDX, ENGINE_IDX } SerializedInfoIndex;

struct CudaDevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  std::string device_name;

  CudaDevice();
  CudaDevice(int64_t gpu_id, nvinfer1::DeviceType device_type);
  CudaDevice(std::string serialized_device_info);
  ~CudaDevice() = default;
  CudaDevice(const CudaDevice& other) = default;
  CudaDevice& operator=(const CudaDevice& other);
  std::string serialize();
  std::string getSMCapability() const;
  friend std::ostream& operator<<(std::ostream& os, const CudaDevice& device);
};

void set_cuda_device(CudaDevice& cuda_device);
// Gets the current active GPU (DLA will not show up through this)
CudaDevice get_current_device();

c10::optional<CudaDevice> get_most_compatible_device(const CudaDevice& target_device);
std::vector<CudaDevice> find_compatible_devices(const CudaDevice& target_device);

struct TRTEngine : torch::CustomClassHolder {
  // Each engine needs it's own runtime object
  std::shared_ptr<nvinfer1::IRuntime> rt;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx;
  std::pair<uint64_t, uint64_t> num_io;
  std::string name;
  std::mutex mu;
  CudaDevice device_info;

  std::unordered_map<uint64_t, uint64_t> in_binding_map;
  std::unordered_map<uint64_t, uint64_t> out_binding_map;

  ~TRTEngine() = default;
  TRTEngine(std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine(std::vector<std::string> serialized_info);
  TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice cuda_device);
  TRTEngine& operator=(const TRTEngine& other);
  std::string to_str() const;
  friend std::ostream& operator<<(std::ostream& os, const TRTEngine& engine);
  // TODO: Implement a call method
  // c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

class DeviceList {
  using DeviceMap = std::unordered_map<int, CudaDevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList();

 public:
  void insert(int device_id, CudaDevice cuda_device);
  CudaDevice find(int device_id);
  DeviceMap get_devices();
  std::string dump_list();
};

DeviceList get_available_device_list();
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
