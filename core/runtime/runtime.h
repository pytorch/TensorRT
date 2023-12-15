#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/runtime/RTDevice.h"
#include "core/runtime/TRTEngine.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "4";
extern bool MULTI_DEVICE_SAFE_MODE;
typedef enum {
  ABI_TARGET_IDX = 0,
  NAME_IDX,
  DEVICE_IDX,
  ENGINE_IDX,
  INPUT_BINDING_NAMES_IDX,
  OUTPUT_BINDING_NAMES_IDX,
  SERIALIZATION_LEN, // NEVER USED FOR DATA, USED TO DETERMINE LENGTH OF SERIALIZED INFO
} SerializedInfoIndex;

c10::optional<RTDevice> get_most_compatible_device(
    const RTDevice& target_device,
    const RTDevice& curr_device = RTDevice());
std::vector<RTDevice> find_compatible_devices(const RTDevice& target_device);

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

void multi_gpu_device_check();

class DeviceList {
  using DeviceMap = std::unordered_map<int, RTDevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList();

 public:
  void insert(int device_id, RTDevice cuda_device);
  RTDevice find(int device_id);
  DeviceMap get_devices();
  std::string dump_list();
};

DeviceList get_available_device_list();
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs();

void set_rt_device(RTDevice& cuda_device);
// Gets the current active GPU (DLA will not show up through this)
RTDevice get_current_device();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
