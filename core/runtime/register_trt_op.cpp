#include "c10/cuda/CUDAStream.h"

#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/torch.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace runtime {

// Checks if the context switch requred for device ID
bool is_switch_required(const CudaDevice& curr_device, const CudaDevice& conf_device) {
  // If SM capability is not the same as configured then switch
  if ((curr_device.major != conf_device.major) || (curr_device.minor != conf_device.minor)) {
    LOG_WARNING(
        "Configured SM capability " << conf_device.getSMCapability()
                                    << " does not match with current device SM capability "
                                    << curr_device.getSMCapability() << " (" << curr_device
                                    << "). Switching device context");
    return true;
  }

  // GPU case
  if (conf_device.device_type == nvinfer1::DeviceType::kGPU) {
    if (curr_device.device_name != conf_device.device_name) {
      LOG_WARNING(
          "Program compiled for " << conf_device.device_name << " but current CUDA device is " << curr_device
                                  << ". Attempting to switch device context for better compatibility");
      return true;
    }
  }

  if (curr_device.id != conf_device.id) {
    LOG_WARNING(
        "Configured Device ID: " << conf_device.id << " is different that current device ID: " << curr_device.id
                                 << ". Moving input tensors to device: " << conf_device.id);
    return true;
  }

  return false;
}

CudaDevice select_cuda_device(const CudaDevice& conf_device) {
  int64_t device_id = -1;
  auto dla_supported = get_dla_supported_SMs();

  auto device_list = get_available_device_list().get_devices();

  CudaDevice new_target_device;

  for (auto device : device_list) {
    auto compute_cap = device.second.getSMCapability();
    // In case of DLA select the DLA supported device ID
    if (conf_device.device_type == nvinfer1::DeviceType::kDLA) {
      if (dla_supported.find(compute_cap) != dla_supported.end() &&
          dla_supported[compute_cap] == device.second.device_name) {
        device_id = device.second.id;
        new_target_device = CudaDevice(device_id, nvinfer1::DeviceType::kDLA);
        break;
      }
    } else if (conf_device.device_type == nvinfer1::DeviceType::kGPU) {
      auto conf_sm = conf_device.getSMCapability();
      if (compute_cap == conf_sm && device.second.device_name == conf_device.device_name) {
        device_id = device.second.id;
        new_target_device = CudaDevice(device_id, nvinfer1::DeviceType::kGPU);
        break;
      }
    } else {
      TRTORCH_THROW_ERROR("Unknown target device type detected from the compiled program (runtime.select_cuda_device)");
      break;
    }
  }

  // REVIEW: THIS DOES NOT LIST DLA PROBABLY, WHICH WE SHOULD
  TRTORCH_CHECK(
      device_id >= 0,
      "No compatible device found on system to run program.\n Program targets "
          << conf_device << "\n Available targets: \n"
          << get_available_device_list().dump_list() << "\n(runtime.select_cuda_device)");
  return new_target_device;
}

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine) {
  LOG_DEBUG("Attempting to run engine (ID: " << compiled_engine->name << ")");

  CudaDevice curr_device = get_current_device();
  LOG_DEBUG("Current Device: " << curr_device);

  if (is_switch_required(curr_device, compiled_engine->device_info)) {
    // Scan through available CUDA devices and set the CUDA device context correctly
    CudaDevice device = select_cuda_device(compiled_engine->device_info);
    set_cuda_device(device);

    std::string target_device = "cuda:" + std::to_string(device.id);

    for (auto& in : inputs) {
      in = in.to(at::kCUDA);
    }
  }

  std::vector<void*> gpu_handles;

  std::vector<at::Tensor> contig_inputs{};
  contig_inputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    uint64_t pyt_idx = compiled_engine->in_binding_map[i];
    TRTORCH_CHECK(
        inputs[pyt_idx].is_cuda(),
        "Expected input tensors to have device cuda, found device " << inputs[pyt_idx].device());
    auto expected_type = util::toATenDType(compiled_engine->exec_ctx->getEngine().getBindingDataType(i));
    TRTORCH_CHECK(
        inputs[pyt_idx].dtype() == expected_type,
        "Expected input tensors to have type " << expected_type << ", found type " << inputs[pyt_idx].dtype());
    auto dims = core::util::toDimsPad(inputs[pyt_idx].sizes(), 1);
    auto shape = core::util::toVec(dims);
    contig_inputs.push_back(inputs[pyt_idx].view(shape).contiguous());
    LOG_DEBUG("Input shape: " << dims);
    compiled_engine->exec_ctx->setBindingDimensions(i, dims);
    gpu_handles.push_back(contig_inputs.back().data_ptr());
  }

  TRTORCH_CHECK(
      compiled_engine->exec_ctx->allInputDimensionsSpecified(), "Not enough inputs provided (runtime.RunCudaEngine)");

  std::vector<at::Tensor> outputs(compiled_engine->num_io.second);
  for (size_t o = inputs.size(); o < (compiled_engine->num_io.first + compiled_engine->num_io.second); o++) {
    uint64_t pyt_idx = compiled_engine->out_binding_map[o];
    auto out_shape = compiled_engine->exec_ctx->getBindingDimensions(o);
    LOG_DEBUG("Output shape: " << out_shape);
    auto dims = core::util::toVec(out_shape);
    auto type = util::toATenDType(compiled_engine->exec_ctx->getEngine().getBindingDataType(o));
    outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());
    gpu_handles.push_back(outputs[pyt_idx].data_ptr());
  }

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());
  compiled_engine->exec_ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

  return outputs;
}

TORCH_LIBRARY(tensorrt, m) {
  m.def("execute_engine", execute_engine);
}

} // namespace runtime
} // namespace core
} // namespace trtorch
