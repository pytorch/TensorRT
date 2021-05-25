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
    LOG_WARNING("Configured SM capability does not match with current device ID. Switching context");
    return true;
  }

  // GPU case
  if (conf_device.device_type == nvinfer1::DeviceType::kGPU) {
    if (curr_device.device_name != conf_device.device_name) {
      LOG_WARNING(
          "TRTEngine compiled for " << conf_device.device_name << " but current CUDA device is "
                                    << curr_device.device_name << ". Switching the device context");
      return true;
    }
  }

  if (curr_device.id != conf_device.id) {
    LOG_WARNING(
        "Configured Device ID: " << conf_device.id << " is different that current device ID: " << curr_device.id
                                 << ". Switching context");
    return true;
  }

  return false;
}

int select_cuda_device(const CudaDevice& conf_device) {
  int device_id = 0;
  int num_devices = 0;
  // SM Compute capability <major,minor> pair
  std::unordered_map<std::string, std::string> dla_supported_SM;

  // Xavier SM Compute Capability
  dla_supported_SM.insert(std::make_pair("7.2", "Xavier"));
  auto status = cudaGetDeviceCount(&num_devices);
  TRTORCH_CHECK((status == cudaSuccess), "Unable to read CUDA capable devices. Return status: " << status);

  cudaDeviceProp device_prop;

  for (int i = 0; i < num_devices; i++) {
    TRTORCH_CHECK(
        (cudaGetDeviceProperties(&device_prop, i) == cudaSuccess),
        "Unable to read CUDA Device Properies for device id: " << i);
    auto compute_cap = std::to_string(device_prop.major) + "." + std::to_string(device_prop.minor);
    std::string device_name{device_prop.name};
    // In case of DLA select the DLA supported device ID
    if (conf_device.device_type == nvinfer1::DeviceType::kDLA) {
      if (dla_supported_SM.find(compute_cap) != dla_supported_SM.end() &&
          dla_supported_SM[compute_cap] == device_name) {
        device_id = i;
        break;
      }
    } else if (conf_device.device_type == nvinfer1::DeviceType::kGPU) {
      auto conf_sm = std::to_string(conf_device.major) + "." + std::to_string(conf_device.minor);
      if (compute_cap == conf_sm && device_name == conf_device.device_name) {
        device_id = i;
        break;
      }
    } else {
      LOG_ERROR("Unkown device type detected from the compiled engine");
      break;
    }
  }
  return device_id;
}

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine) {
  LOG_DEBUG("Attempting to run engine (ID: " << compiled_engine->name << ")");

  CudaDevice curr_device;
  get_cuda_device(curr_device);

  if (is_switch_required(curr_device, compiled_engine->device_info)) {
    // Scan through available CUDA devices and set the CUDA device context correctly
    CudaDevice device{.id = select_cuda_device(compiled_engine->device_info)};
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
