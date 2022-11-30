#pragma once
#include <string>
#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct RTDevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  std::string device_name;

  RTDevice();
  RTDevice(int64_t gpu_id, nvinfer1::DeviceType device_type);
  RTDevice(std::string serialized_device_info);
  ~RTDevice() = default;
  RTDevice(const RTDevice& other) = default;
  RTDevice& operator=(const RTDevice& other);
  std::string serialize();
  std::string getSMCapability() const;
  friend std::ostream& operator<<(std::ostream& os, const RTDevice& device);
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
