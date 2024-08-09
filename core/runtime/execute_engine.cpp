#include "c10/cuda/CUDAGuard.h"
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

RTDevice select_rt_device(const RTDevice& engine_device, const RTDevice& curr_device, bool hardware_compatible) {
  auto new_target_device_opt = get_most_compatible_device(engine_device, curr_device, hardware_compatible);

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

bool _cudagraphs_validate_shapes(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine) {
  // Validate whether the current input shapes to the engine
  // invalidate the existing cudagraphs object

  // Populate the shape key for the inputs
  // x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
  std::stringstream new_shape_key_ss;
  for (auto input : inputs) {
    new_shape_key_ss << "(";
    auto sizes = input.sizes();
    auto rank = input.sizes().size();
    for (auto i = 0; i < rank; i++) {
      new_shape_key_ss << sizes[i];
      // For all but the final dimension in the shape key, add comma separator
      if (i < rank - 1) {
        new_shape_key_ss << ",";
      }
    }
    new_shape_key_ss << ")";
  }

  auto new_shape_key = new_shape_key_ss.str();

  // Compare the shape key to the original key and invalidate shapes if they do not match
  if (new_shape_key != compiled_engine->shape_key) {
    LOG_DEBUG("Resetting Cudagraph on New Shape Key " << new_shape_key);
    compiled_engine->shape_key = new_shape_key;
    compiled_engine->cudagraph.reset();
    return false;
  }

  return true;
}

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine) {
  LOG_DEBUG(
      "Attempting to run engine (ID: " << compiled_engine->name
                                       << "); Hardware Compatible: " << compiled_engine->hardware_compatible);

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

  // Whether cudagraphs needs to record the graph on this pass
  bool need_cudagraphs_record = (CUDAGRAPHS_MODE && !_cudagraphs_validate_shapes(inputs, compiled_engine));

  // this is a buffer to store shape tensor input addresses throughout the runtime scope
  std::list<std::vector<int64_t>> inputShapeTensorValues;

  // Intialize outputs to be available throughout the succeeding scopes
  std::vector<at::Tensor> outputs(compiled_engine->num_io.second);

  // If not in cudagraphs mode or a new cudagraphs recording is needed
  // proceed with input validation and assignment of new I/O pointers for TRT
  if (!CUDAGRAPHS_MODE || need_cudagraphs_record) {
    if (MULTI_DEVICE_SAFE_MODE) {
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
        RTDevice device =
            select_rt_device(compiled_engine->device_info, curr_device, compiled_engine->hardware_compatible);
        set_rt_device(device);

        // Update active stream based on new device
        auto current_stream = c10::cuda::getCurrentCUDAStream(device.id);
        if (current_stream == c10::cuda::getDefaultCUDAStream(device.id)) {
          compiled_engine->active_stream = c10::cuda::getStreamFromPool(false, device.id);
          c10::cuda::setCurrentCUDAStream(compiled_engine->active_stream);
        } else {
          compiled_engine->active_stream = current_stream;
        }

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
        auto dims = core::util::toDims(inputs[i].sizes());
        auto shape = core::util::toVec(dims);
        LOG_DEBUG("Input Name: " << name << " Shape: " << dims);
        at::Tensor contig_input;

        if (compiled_engine->cuda_engine->isShapeInferenceIO(name.c_str())) {
          // Shape tensor inputs are casted to int64 explicitly.
          // Refer to
          // https://github.com/NVIDIA/TensorRT/blob/d2f4ef789a9a6ffdf37b55c3f81b486225f6b380/samples/common/sampleInference.cpp#L435
          auto input_cpu = inputs[i].clone().contiguous().cpu().to(torch::kInt64);
          std::vector<int64_t> inputs_cpu_vec(
              input_cpu.data_ptr<int64_t>(), input_cpu.data_ptr<int64_t>() + input_cpu.numel());
          inputShapeTensorValues.emplace_back(inputs_cpu_vec);
          TORCHTRT_CHECK(
              compiled_engine->exec_ctx->setTensorAddress(name.c_str(), inputShapeTensorValues.back().data()),
              "Error while setting the tensor address for shape inputs");
          compiled_engine->input_buffers[i] = input_cpu;
        } else {
          // If in cudagraphs mode, the inputs must be cloned since the memory will be reused
          // in subsequent replays of the graph
          if (CUDAGRAPHS_MODE) {
            contig_input = inputs[i].view(shape).contiguous().clone();
            compiled_engine->input_buffers[i] = contig_input;
          } else {
            contig_input = inputs[i].view(shape).contiguous();
          }
          TORCHTRT_CHECK(
              compiled_engine->exec_ctx->setInputShape(name.c_str(), dims), "Error while setting the input shape");
          TORCHTRT_CHECK(
              compiled_engine->exec_ctx->setTensorAddress(name.c_str(), contig_input.data_ptr()),
              "Error while setting the input tensor address for inputs");
        }
      }

      // Check if input shapes can be inferred.
      int32_t const io_size{compiled_engine->cuda_engine->getNbIOTensors()};
      std::vector<char const*> names(io_size);
      int32_t const nbNames = compiled_engine->exec_ctx->inferShapes(names.size(), names.data());
      TORCHTRT_CHECK(
          nbNames == 0,
          "The shapes of the inputs: "
              << names
              << " cannot be inferred. This could happen if the input tensor addresses/shapes haven't been configured correctly");
    }

    std::unique_ptr<torch::autograd::profiler::RecordProfile> output_profiler_guard;
    if (compiled_engine->profile_execution) {
      output_profiler_guard =
          std::make_unique<torch::autograd::profiler::RecordProfile>(compiled_engine->output_profile_path);
    }

    for (auto output_indices : compiled_engine->out_binding_map) {
      // out_binding_map stores TRT_IDX: PYT_IDX
      auto pyt_idx = output_indices.second;

      std::string name = compiled_engine->out_binding_names[pyt_idx];
      auto out_shape = compiled_engine->exec_ctx->getTensorShape(name.c_str());
      LOG_DEBUG("Output Name: " << name << " Shape: " << out_shape);
      auto dims = core::util::toVec(out_shape);
      auto type = util::TRTDataTypeToScalarType(compiled_engine->exec_ctx->getEngine().getTensorDataType(name.c_str()));
      outputs[pyt_idx] = std::move(at::empty(dims, {at::kCUDA}).to(type).contiguous());

      // In cudagraphs mode, the allocated output buffers are stored for reuse
      if (CUDAGRAPHS_MODE) {
        compiled_engine->output_buffers[pyt_idx] = outputs[pyt_idx];
      }
      TORCHTRT_CHECK(
          compiled_engine->exec_ctx->setTensorAddress(name.c_str(), outputs[pyt_idx].data_ptr()),
          "Error while setting the output tensor address");
    }
  }

  std::unique_ptr<torch::autograd::profiler::RecordProfile> enqueue_profiler_guard;

  // nvinfer1::IExecutionContext::enqueue is not thread safe and we need a mutex for it.
  std::unique_lock<std::mutex> lock(compiled_engine->mu);

  if (!CUDAGRAPHS_MODE) {
    // If not in cudagraphs mode, proceed with enqueueV3 as normal
    compiled_engine->exec_ctx->enqueueV3(compiled_engine->active_stream);
  } else if (need_cudagraphs_record) {
    // If cudagraphs needs to record a graph, capture the enqueueV3 call in a graph

    // Cudagraphs cannot record on the current stream, so use an alternate
    c10::cuda::CUDAStream recording_stream = c10::cuda::getStreamFromPool(false, inputs[0].device().index());
    c10::cuda::CUDAStreamGuard guard(recording_stream);

    compiled_engine->exec_ctx->enqueueV3(recording_stream);
    recording_stream.synchronize();

    compiled_engine->cudagraph.capture_begin();
    compiled_engine->exec_ctx->enqueueV3(recording_stream);
    compiled_engine->cudagraph.capture_end();

    // Reset the stream to its original setting
    guard.reset_stream(guard.original_stream());

  } else {
    // If the cudagraph has already been recorded, copy the input buffers and replay it
    for (auto i = 0; i < inputs.size(); i++) {
      compiled_engine->input_buffers[i].copy_(inputs[i], true);
    }
    compiled_engine->cudagraph.replay();
  }

  std::vector<at::Tensor> model_outputs(compiled_engine->num_io.second);

  // In cudagraphs mode, the output buffers can be reused, so they must
  // be cloned before providing them to the user to avoid data corruption
  if (CUDAGRAPHS_MODE) {
    for (auto i = 0; i < compiled_engine->output_buffers.size(); i++) {
      model_outputs[i] = compiled_engine->output_buffers[i].clone();
    }
  } else {
    model_outputs = outputs;
  }

  if (compiled_engine->profile_execution) {
    LOG_INFO(std::endl << *compiled_engine->trt_engine_profiler);
    dump_trace(compiled_engine->trt_engine_profile_path, *compiled_engine->trt_engine_profiler);
    compiled_engine->dump_engine_layer_info();
  }

  return model_outputs;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
