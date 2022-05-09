#include "c10/cuda/CUDAStream.h"

#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/torch.h"

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// Checks if the context switch requred for device ID
bool is_switch_required(const CudaDevice& curr_device, const CudaDevice& engine_device) {
  // If SM capability is not the same as configured then switch
  if ((curr_device.major != engine_device.major) || (curr_device.minor != engine_device.minor)) {
    LOG_WARNING(
        "Configured SM capability " << engine_device.getSMCapability()
                                    << " does not match with current device SM capability "
                                    << curr_device.getSMCapability() << " (" << curr_device
                                    << "). Switching device context");
    return true;
  }

  // GPU case
  if (engine_device.device_type == nvinfer1::DeviceType::kGPU) {
    if (curr_device.device_name != engine_device.device_name) {
      LOG_WARNING(
          "Program compiled for " << engine_device.device_name << " but current CUDA device is " << curr_device
                                  << ". Attempting to switch device context for better compatibility");
      return true;
    }
  }

  if (curr_device.id != engine_device.id) {
    LOG_WARNING(
        "Configured Device ID: " << engine_device.id << " is different that current device ID: " << curr_device.id
                                 << ". Moving input tensors to device: " << engine_device.id);
    return true;
  }

  return false;
}

CudaDevice select_cuda_device(const CudaDevice& engine_device) {
  auto new_target_device_opt = get_most_compatible_device(engine_device);

  // REVIEW: THIS DOES NOT LIST DLA PROBABLY, WHICH WE SHOULD
  // TODO: I think this logic could be way simpler at execution time since if the tensors arent on the right
  // device, its not going to run. We should just set device to engine device and maybe reset and memcpy tensors
  // back to orginal device if needed.
  TORCHTRT_CHECK(
      new_target_device_opt,
      "No compatible device found on system to run program.\n Program targets "
          << engine_device << "\n Available targets: \n"
          << get_available_device_list().dump_list() << "\n(runtime.select_cuda_device)");
  return new_target_device_opt.value();
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
      in = in.to(torch::Device(target_device));
    }
  }

  std::vector<void*> gpu_handles;

  std::vector<at::Tensor> contig_inputs{};
  contig_inputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    uint64_t pyt_idx = compiled_engine->in_binding_map[i];
    TORCHTRT_CHECK(
        inputs[pyt_idx].is_cuda(),
        "Expected input tensors to have device cuda, found device " << inputs[pyt_idx].device());
    auto expected_type = util::TRTDataTypeToScalarType(compiled_engine->exec_ctx->getEngine().getBindingDataType(i));
    TORCHTRT_CHECK(
        inputs[pyt_idx].dtype() == expected_type,
        "Expected input tensors to have type " << expected_type << ", found type " << inputs[pyt_idx].dtype());
    auto dims = core::util::toDimsPad(inputs[pyt_idx].sizes(), 1);
    auto shape = core::util::toVec(dims);
    contig_inputs.push_back(inputs[pyt_idx].view(shape).contiguous());
    LOG_DEBUG("Input shape: " << dims);
    compiled_engine->exec_ctx->setBindingDimensions(i, dims);
    gpu_handles.push_back(contig_inputs.back().data_ptr());
  }

  TORCHTRT_CHECK(
      compiled_engine->exec_ctx->allInputDimensionsSpecified(), "Not enough inputs provided (runtime.RunCudaEngine)");

  std::vector<at::Tensor> outputs(compiled_engine->num_io.second);
  for (size_t o = inputs.size(); o < (compiled_engine->num_io.first + compiled_engine->num_io.second); o++) {
    uint64_t pyt_idx = compiled_engine->out_binding_map[o];
    auto out_shape = compiled_engine->exec_ctx->getBindingDimensions(o);
    LOG_DEBUG("Output shape: " << out_shape);
    auto dims = core::util::toVec(out_shape);
    auto type = util::TRTDataTypeToScalarType(compiled_engine->exec_ctx->getEngine().getBindingDataType(o));
    outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());
    gpu_handles.push_back(outputs[pyt_idx].data_ptr());
  }

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());

  // nvinfer1::IExecutionContext::enqueue is not thread safe and we need a mutex for it.
  std::unique_lock<std::mutex> lock(compiled_engine->mu);
  compiled_engine->exec_ctx->enqueueV2(gpu_handles.data(), stream, nullptr);

  return outputs;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
