#pragma once
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace trtorch {
namespace core {
namespace runtime {

using EngineID = int64_t;

typedef enum {
  DeviceIdx = 0,
  EngineIdx
}SerializedInfoIndex;

struct CudaDevice {
  int64_t id; // CUDA device id
  int64_t major; // CUDA compute major version
  int64_t minor; // CUDA compute minor version
  nvinfer1::DeviceType device_type;
  size_t device_name_len;
  std::string device_name;

  int64_t get_id(void) {
	  return this->id;
  }

  void set_id(int64_t gpu_id) {
	  this->id = gpu_id;
  }

  int64_t get_major(void) {
	  return major;
  }

  void set_major(int64_t major_version) {
	  major = major_version;
  }

  int64_t get_minor(void) {
	  return minor;
  }

  void set_minor(int64_t minor_version) {
	  minor = minor_version;
  }

  nvinfer1::DeviceType get_device_type(void) {
	  return device_type;
  }

  void set_device_type(nvinfer1::DeviceType device_type) {
	  device_type = device_type;
  }

  std::string get_device_name(void) {
	  return device_name;
  }

  void set_device_name(std::string& name) {
	  device_name = name;
  }

  size_t get_device_name_len(void) {
	  return device_name_len;
  }

  void set_device_name_len(size_t size) {
	  device_name_len = size;
  }
};

void set_cuda_device(CudaDevice& cuda_device);
void get_cuda_device(CudaDevice& cuda_device);

std::string serialize_device(CudaDevice& cuda_device);
CudaDevice deserialize_device(std::string device_info);

CudaDevice get_device_info(int64_t gpu_id, nvinfer1::DeviceType device_type);

struct TRTEngine : torch::CustomClassHolder {
	// Each engine needs it's own runtime object
	nvinfer1::IRuntime* rt;
	nvinfer1::ICudaEngine* cuda_engine;
	nvinfer1::IExecutionContext* exec_ctx;
	std::pair<uint64_t, uint64_t> num_io;
	EngineID id;
	std::string name;
	CudaDevice device_info;
	util::logging::TRTorchLogger logger;

	std::unordered_map<uint64_t, uint64_t> in_binding_map;
	std::unordered_map<uint64_t, uint64_t> out_binding_map;

	~TRTEngine();
	TRTEngine(std::string serialized_engine);
	TRTEngine(std::vector<std::string> serialized_info);
	TRTEngine(std::string mod_name, std::string serialized_engine, CudaDevice device);
	TRTEngine& operator=(const TRTEngine& other);
	// TODO: Implement a call method
	// c10::List<at::Tensor> Run(c10::List<at::Tensor> inputs);
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

} // namespace runtime
} // namespace core
} // namespace trtorch
