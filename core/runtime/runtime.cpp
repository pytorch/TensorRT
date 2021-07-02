#include "cuda_runtime.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace runtime {

void set_cuda_device(CudaDevice& cuda_device) {
  TRTORCH_CHECK(
      (cudaSetDevice(cuda_device.id) == cudaSuccess), "Unable to set device: " << cuda_device << "as active device");
}

CudaDevice get_current_device() {
  int device = -1;
  TRTORCH_CHECK(
      (cudaGetDevice(reinterpret_cast<int*>(&device)) == cudaSuccess),
      "Unable to get current device (runtime.get_current_device)");

  int64_t device_id = static_cast<int64_t>(device);

  return CudaDevice(device_id, nvinfer1::DeviceType::kGPU);
}

std::string serialize_device(CudaDevice& cuda_device) {
  return cuda_device.serialize();
}

CudaDevice deserialize_device(std::string device_info) {
  return CudaDevice(device_info);
}

namespace {
static DeviceList cuda_device_list;
}

DeviceList get_available_device_list() {
  return cuda_device_list;
}

// SM Compute capability <Compute Capability, Device Name> map
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs() {
  // Xavier SM Compute Capability
  static std::unordered_map<std::string, std::string> dla_supported_SM = {{"7.2", "Xavier"}};
  return dla_supported_SM;
}

} // namespace runtime
} // namespace core
} // namespace trtorch
