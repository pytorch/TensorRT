#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "NvInfer.h"
#include "core/runtime/Platform.h"
#include "core/runtime/RTDevice.h"
#include "core/runtime/TRTEngine.h"
#include "core/util/prelude.h"
#include "torch/csrc/stable/library.h"
#include "torch/csrc/stable/tensor_struct.h"
#include "torch/csrc/stable/ops.h"
#include "torch/csrc/stable/stableivalue_conversions.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/macros/Macros.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "7";
extern bool MULTI_DEVICE_SAFE_MODE;

typedef enum {
  STANDARD = 0,
  SUBGRAPH_CUDAGRAPHS,
  WHOLE_GRAPH_CUDAGRAPHS,
} CudaGraphsMode;

extern CudaGraphsMode CUDAGRAPHS_MODE;

typedef enum {
  ABI_TARGET_IDX = 0,
  NAME_IDX,
  DEVICE_IDX,
  ENGINE_IDX,
  INPUT_BINDING_NAMES_IDX,
  OUTPUT_BINDING_NAMES_IDX,
  HW_COMPATIBLE_IDX,
  SERIALIZED_METADATA_IDX,
  TARGET_PLATFORM_IDX,
  REQUIRES_OUTPUT_ALLOCATOR_IDX,
  SERIALIZATION_LEN, // NEVER USED FOR DATA, USED TO DETERMINE LENGTH OF SERIALIZED INFO
} SerializedInfoIndex;

std::string base64_encode(const std::string& in);
std::string base64_decode(const std::string& in);
std::string serialize_bindings(const std::vector<std::string>& bindings);

c10::optional<RTDevice> get_most_compatible_device(
    const RTDevice& target_device,
    const RTDevice& curr_device = RTDevice(),
    bool hardware_compatible = false);
std::vector<RTDevice> find_compatible_devices(const RTDevice& target_device, bool hardware_compatible);

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);
void execute_engine_boxed(StableIValue* stack, uint64_t num_args, uint64_t num_outputs);

void multi_gpu_device_check();

bool get_multi_device_safe_mode();

void set_multi_device_safe_mode(bool multi_device_safe_mode);

CudaGraphsMode get_cudagraphs_mode();

void set_cudagraphs_mode(CudaGraphsMode cudagraphs_mode);

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
