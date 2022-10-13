#pragma once
#include <string>
#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct CUDADevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  std::string device_name;

  CUDADevice();
  CUDADevice(int64_t gpu_id, nvinfer1::DeviceType device_type);
  CUDADevice(std::string serialized_device_info);
  ~CUDADevice() = default;
  CUDADevice(const CUDADevice& other) = default;
  CUDADevice& operator=(const CUDADevice& other);
  std::string serialize();
  std::string getSMCapability() const;
  friend std::ostream& operator<<(std::ostream& os, const CUDADevice& device);
};

void set_cuda_device(CUDADevice& cuda_device);
// Gets the current active GPU (DLA will not show up through this)
CUDADevice get_current_device();

} // namespace torch_tensorrt
} // namespace core
} // namespace runtime
