#include "c10/cuda/CUDAStream.h"

#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/torch.h"

#include "core/runtime/TRTEngineProfiler.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// Checks if the context switch requred for device ID
bool is_switch_required(const RTDevice& curr_device, const RTDevice& engine_device) {
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

RTDevice select_rt_device(const RTDevice& engine_device) {
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

  if (compiled_engine->profile_execution) {
    std::stringstream ss;
    ss << "Execution profiling is enabled, find results here:" << std::endl;
    compiled_engine->set_profiling_paths();
    ss << "  Device selection profile: " << compiled_engine->device_profile_path << std::endl;
    ss << "  Input packing profile: " << compiled_engine->input_profile_path << std::endl;
    ss << "  Output packing profile: " << compiled_engine->output_profile_path << std::endl;
    ss << "  TRT enqueue profile: " << compiled_engine->enqueue_profile_path << std::endl;
    ss << "  Engine execution profile: " << compiled_engine->trt_engine_profile_path << std::endl;
    auto log_info = ss.str();
    LOG_INFO("" << log_info);
  }

  {
    std::unique_ptr<torch::autograd::profiler::RecordProfile> device_profiler_guard;
    if (compiled_engine->profile_execution) {
      device_profiler_guard =
          std::make_unique<torch::autograd::profiler::RecordProfile>(compiled_engine->device_profile_path);
    }

    RTDevice curr_device = get_current_device();
    LOG_DEBUG("Current Device: " << curr_device);

    // Generic Target Device Prefix
    std::string target_device = "cuda:";

    if (is_switch_required(curr_device, compiled_engine->device_info)) {
      // Scan through available CUDA devices and set the CUDA device context correctly
      RTDevice device = select_rt_device(compiled_engine->device_info);
      set_rt_device(device);

      // Target device is new device
      target_device += std::to_string(device.id);

      for (auto& in : inputs) {
        in = in.to(torch::Device(target_device));
      }
    } else {
      // Target device is current device
      target_device += std::to_string(curr_device.id);
    }

    // For each input, ensure its current device is the desired target device
    for (size_t i = 0; i < inputs.size(); i++) {
      at::Tensor* in = &inputs[i];
      std::string current_tensor_device = in->device().str();

      // If current device string does not match target device, display warning and move tensor accordingly
      if (current_tensor_device != target_device) {
        LOG_WARNING(
            "Input " << i << " of engine " << compiled_engine->name << " was found to be on " << current_tensor_device
                     << " but should be on " << target_device << ". This tensor is being moved by the runtime but "
                     << "for performance considerations, ensure your inputs are all on GPU "
                     << "and open an issue here (https://github.com/pytorch/TensorRT/issues) if this "
                     << "warning persists.");
        *in = in->to(torch::Device(target_device));
      }
    }
  }

  {
    std::unique_ptr<torch::autograd::profiler::RecordProfile> input_profiler_guard;
    if (compiled_engine->profile_execution) {
      input_profiler_guard =
          std::make_unique<torch::autograd::profiler::RecordProfile>(compiled_engine->input_profile_path);
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      std::string name = compiled_engine->in_binding_names[i];
      TORCHTRT_CHECK(
          inputs[i].is_cuda(), "Expected input tensors to have device cuda, found device " << inputs[i].device());
      auto expected_type =
          util::TRTDataTypeToScalarType(compiled_engine->exec_ctx->getEngine().getTensorDataType(name.c_str()));
      TORCHTRT_CHECK(
          inputs[i].dtype() == expected_type,
          "Expected input tensors to have type " << expected_type << ", found type " << inputs[i].dtype());
      auto dims = core::util::toDimsPad(inputs[i].sizes(), 1);
      auto shape = core::util::toVec(dims);
      LOG_DEBUG("Input Name: " << name << " Shape: " << dims);
      compiled_engine->exec_ctx->setInputShape(name.c_str(), dims);
      compiled_engine->exec_ctx->setTensorAddress(name.c_str(), inputs[i].view(shape).contiguous().data_ptr());
    }

    TORCHTRT_CHECK(
        compiled_engine->exec_ctx->allInputShapesSpecified(), "Not enough inputs provided (runtime.RunCudaEngine)");
  }

  std::vector<at::Tensor> outputs(compiled_engine->num_io.second);
  {
    std::unique_ptr<torch::autograd::profiler::RecordProfile> output_profiler_guard;
    if (compiled_engine->profile_execution) {
      output_profiler_guard =
          std::make_unique<torch::autograd::profiler::RecordProfile>(compiled_engine->output_profile_path);
    }

    for (size_t o = inputs.size(); o < (compiled_engine->num_io.first + compiled_engine->num_io.second); o++) {
      uint64_t pyt_idx = compiled_engine->out_binding_map[o];
      std::string name = compiled_engine->out_binding_names[pyt_idx];
      auto out_shape = compiled_engine->exec_ctx->getTensorShape(name.c_str());
      LOG_DEBUG("Output Name: " << name << " Shape: " << out_shape);
      auto dims = core::util::toVec(out_shape);
      auto type = util::TRTDataTypeToScalarType(compiled_engine->exec_ctx->getEngine().getTensorDataType(name.c_str()));
      outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());
      compiled_engine->exec_ctx->setTensorAddress(name.c_str(), outputs[pyt_idx].data_ptr());
    }
  }

  {
    std::unique_ptr<torch::autograd::profiler::RecordProfile> enqueue_profiler_guard;
    if (compiled_engine->profile_execution) {
      enqueue_profiler_guard =
          std::make_unique<torch::autograd::profiler::RecordProfile>(compiled_engine->enqueue_profile_path);
    }
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());

    // nvinfer1::IExecutionContext::enqueue is not thread safe and we need a mutex for it.
    std::unique_lock<std::mutex> lock(compiled_engine->mu);
    compiled_engine->exec_ctx->enqueueV3(stream);
    if (compiled_engine->profile_execution) {
      LOG_INFO(std::endl << *compiled_engine->trt_engine_profiler);
      dump_trace(compiled_engine->trt_engine_profile_path, *compiled_engine->trt_engine_profiler);
      compiled_engine->dump_engine_layer_info();
    }
  }
  return outputs;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
