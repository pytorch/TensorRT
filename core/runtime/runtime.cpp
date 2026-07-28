#include "cuda_runtime.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

bool MULTI_DEVICE_SAFE_MODE = false;
CudaGraphsMode CUDAGRAPHS_MODE = STANDARD;

c10::optional<RTDevice> get_most_compatible_device(
    const RTDevice& target_device,
    const RTDevice& curr_device,
    bool hardware_compatible) {
  LOG_DEBUG("Target Device: " << target_device);
  auto device_options = find_compatible_devices(target_device, hardware_compatible);
  RTDevice current_device;
  if (current_device.id == -1) {
    current_device = get_current_device();
  } else {
    current_device = curr_device;
  }

  if (device_options.size() == 0) {
    return {};
  } else if (device_options.size() == 1) {
    return {device_options[0]};
  }

  RTDevice best_match;
  std::stringstream dev_list;
  dev_list << "[" << std::endl;
  for (auto device : device_options) {
    dev_list << "    " << device << ',' << std::endl;
    // If the model is hardware compatible, any compatible device should be valid
    if ((device.device_name == target_device.device_name) || hardware_compatible) {
      // First priority is selecting a candidate which agrees with the current device ID
      // If such a device is found, we can select it and break out of the loop
      if (device.id == current_device.id) {
        best_match = device;
        break;
      }
      // Second priority is selecting a candidate which agrees with the target device ID
      // At deserialization time, the current device and target device may not agree
      else if (device.id == target_device.id) {
        best_match = device;
      }
      // If no such GPU ID is found, select the first available candidate GPU
      else if (best_match.device_name != target_device.device_name) {
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

std::vector<RTDevice> find_compatible_devices(const RTDevice& target_device, bool hardware_compatible) {
  auto dla_supported = get_dla_supported_SMs();
  auto device_list = get_available_device_list().get_devices();

  std::vector<RTDevice> compatible_devices;

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
      // If hardware compatibility mode is enabled and the SM is at least 80, device is valid
      if ((poss_dev_cc == target_dev_cc) || (hardware_compatible && std::stoi(poss_dev_cc) >= 8)) {
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

void set_rt_device(RTDevice& cuda_device) {
  TORCHTRT_CHECK(
      (cudaSetDevice(cuda_device.id) == cudaSuccess), "Unable to set device: " << cuda_device << "as active device");
  LOG_DEBUG("Setting " << cuda_device << " as active device");
}

RTDevice get_current_device() {
  int device = -1;
  TORCHTRT_CHECK(
      (cudaGetDevice(reinterpret_cast<int*>(&device)) == cudaSuccess),
      "Unable to get current device (runtime.get_current_device)");

  int64_t device_id = static_cast<int64_t>(device);

  return RTDevice(device_id, nvinfer1::DeviceType::kGPU);
}

void multi_gpu_device_check() {
  // If multi-device safe mode is disabled and more than 1 device is registered on the machine, warn user
  if (!(MULTI_DEVICE_SAFE_MODE) && get_available_device_list().get_devices().size() > 1) {
    LOG_WARNING(
        "Detected this engine is being instantitated in a multi-GPU system with "
        << "multi-device safe mode disabled. For more on the implications of this "
        << "as well as workarounds, see the linked documentation "
        << "(https://pytorch.org/TensorRT/user_guide/runtime.html#multi-device-safe-mode)");
  }
}

bool get_multi_device_safe_mode() {
  return MULTI_DEVICE_SAFE_MODE;
}

void set_multi_device_safe_mode(bool multi_device_safe_mode) {
  MULTI_DEVICE_SAFE_MODE = multi_device_safe_mode;
}

CudaGraphsMode get_cudagraphs_mode() {
  return CUDAGRAPHS_MODE;
}

void set_cudagraphs_mode(CudaGraphsMode cudagraphs_mode) {
  CUDAGRAPHS_MODE = cudagraphs_mode;
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
