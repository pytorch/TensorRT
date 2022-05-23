#include "cuda_runtime.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

c10::optional<CudaDevice> get_most_compatible_device(const CudaDevice& target_device) {
  LOG_DEBUG("Target Device: " << target_device);
  auto device_options = find_compatible_devices(target_device);
  if (device_options.size() == 0) {
    return {};
  } else if (device_options.size() == 1) {
    return {device_options[0]};
  }

  CudaDevice best_match;
  std::stringstream dev_list;
  dev_list << "[" << std::endl;
  for (auto device : device_options) {
    dev_list << "    " << device << ',' << std::endl;
    if (device.device_name == target_device.device_name && best_match.device_name != target_device.device_name) {
      best_match = device;
    } else if (device.device_name == target_device.device_name && best_match.device_name == target_device.device_name) {
      if (device.id == target_device.id && best_match.id != target_device.id) {
        best_match = device;
      }
    }
  }
  dev_list << ']';
  LOG_DEBUG("Compatible device options: " << dev_list.str());

  if (best_match.id == -1) {
    LOG_DEBUG("No valid device options");
    return {};
  } else {
    LOG_DEBUG("Selected: " << best_match);
    return {best_match};
  }
}

std::vector<CudaDevice> find_compatible_devices(const CudaDevice& target_device) {
  auto dla_supported = get_dla_supported_SMs();
  auto device_list = get_available_device_list().get_devices();

  std::vector<CudaDevice> compatible_devices;

  for (auto device : device_list) {
    auto poss_dev_cc = device.second.getSMCapability();
    if (target_device.device_type == nvinfer1::DeviceType::kDLA) {
      if (dla_supported.find(poss_dev_cc) != dla_supported.end() &&
          dla_supported[poss_dev_cc] == target_device.device_name) {
        compatible_devices.push_back(device.second);
      }
    } else if (target_device.device_type == nvinfer1::DeviceType::kGPU) {
      auto target_dev_cc = target_device.getSMCapability();
      // If the SM Capabilities match, should be good enough to run
      if (poss_dev_cc == target_dev_cc) {
        compatible_devices.push_back(device.second);
      }
    } else {
      TORCHTRT_THROW_ERROR(
          "Unknown target device type detected from the compiled program (runtime.find_compatible_devices)");
      break;
    }
  }
  return compatible_devices;
}

void set_cuda_device(CudaDevice& cuda_device) {
  TORCHTRT_CHECK(
      (cudaSetDevice(cuda_device.id) == cudaSuccess), "Unable to set device: " << cuda_device << "as active device");
  LOG_DEBUG("Setting " << cuda_device << " as active device");
}

CudaDevice get_current_device() {
  int device = -1;
  TORCHTRT_CHECK(
      (cudaGetDevice(reinterpret_cast<int*>(&device)) == cudaSuccess),
      "Unable to get current device (runtime.get_current_device)");

  int64_t device_id = static_cast<int64_t>(device);

  return CudaDevice(device_id, nvinfer1::DeviceType::kGPU);
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
  static std::unordered_map<std::string, std::string> dla_supported_SM = {{"7.2", "Xavier"}, {"8.7", "Orin"}};
  return dla_supported_SM;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
