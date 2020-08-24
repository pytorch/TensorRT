#pragma once
#include <cuda_runtime.h>
#include <string>

namespace trtorch
{
namespace core
{
namespace util
{
    typedef struct {
        int id;         // CUDA device id
	int major;      // CUDA compute major version
	int minor;      // CUDA compute minor version
    }CudaDevice;

    cudaError_t set_cuda_device(CudaDevice& cuda_device);
    cudaError_t get_cuda_device(CudaDevice& cuda_device);

    std::string serialize_device(CudaDevice& cuda_device);
    CudaDevice deserialize_device(std::string device_info);
}
}
}
