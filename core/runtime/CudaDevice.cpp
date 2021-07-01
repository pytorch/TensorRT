#include "cuda_runtime.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace runtime {

const std::string DEVICE_INFO_DELIM = "%";

typedef enum { ID_IDX = 0, SM_MAJOR_IDX, SM_MINOR_IDX, DEVICE_TYPE_IDX, DEVICE_NAME_IDX } SerializedDeviceInfoIndex;

CudaDevice::CudaDevice() : id{-1}, major{-1}, minor{-1}, device_type{nvinfer1::DeviceType::kGPU} {}

CudaDevice::CudaDevice(int64_t gpu_id, nvinfer1::DeviceType device_type) {
  CudaDevice cuda_device;
  cudaDeviceProp device_prop;

  // Device ID
  this->id = gpu_id;

  // Get Device Properties
  cudaGetDeviceProperties(&device_prop, gpu_id);

  // Compute capability major version
  this->major = device_prop.major;

  // Compute capability minor version
  this->minor = device_prop.minor;

  std::string device_name(device_prop.name);

  // Set Device name
  this->device_name = device_name;

  // Set Device name len for serialization/deserialization
  this->device_name_len = device_name.size();

  // Set Device Type
  this->device_type = device_type;
}

// NOTE: Serialization Format for Device Info:
// id%major%minor%(enum)device_type%device_name

CudaDevice::CudaDevice(std::string device_info) {
  LOG_DEBUG("Deserializing Device Info: " << device_info);

  std::vector<std::string> tokens;
  int64_t start = 0;
  int64_t end = device_info.find(DEVICE_INFO_DELIM);

  while (end != -1) {
    tokens.push_back(device_info.substr(start, end - start));
    start = end + DEVICE_INFO_DELIM.size();
    end = device_info.find(DEVICE_INFO_DELIM, start);
  }
  tokens.push_back(device_info.substr(start, end - start));

  TRTORCH_CHECK(tokens.size() == DEVICE_NAME_IDX + 1, "Unable to deserializable program target device infomation");

  id = std::stoi(tokens[ID_IDX]);
  major = std::stoi(tokens[SM_MAJOR_IDX]);
  minor = std::stoi(tokens[SM_MINOR_IDX]);
  device_type = (nvinfer1::DeviceType)(std::stoi(tokens[DEVICE_TYPE_IDX]));
  device_name = tokens[DEVICE_NAME_IDX];

  LOG_DEBUG("Deserialized Device Info: " << *this);
}

std::string CudaDevice::serialize() {
  std::stringstream ss;
  // clang-format off
  ss << id << DEVICE_INFO_DELIM \
     << major << DEVICE_INFO_DELIM \
     << minor << DEVICE_INFO_DELIM \
     << (int64_t) device_type << DEVICE_INFO_DELIM
     << device_name;
  // clang-format on

  std::string serialized_device_info = ss.str();

  LOG_DEBUG("Serialized Device Info: " << serialized_device_info);

  return serialized_device_info;
}

std::string CudaDevice::getSMCapability() const {
  std::stringstream ss;
  ss << major << "." << minor;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const CudaDevice& device) {
  os << "Device(ID: " << device.id << ", Name: " << device.device_name << ", SM Capability: " << device.major << '.'
     << device.minor << ", Type: " << device.device_type << ')';
  return os;
}

} // namespace runtime
} // namespace core
} // namespace trtorch
