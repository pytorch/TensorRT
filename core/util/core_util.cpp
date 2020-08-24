#include <cuda_runtime.h>

#include "core/util/core_util.h"
#include "core/util/macros.h"
#include <cstring>

namespace trtorch
{
namespace core
{
namespace util
{
    cudaError_t set_cuda_device(CudaDevice& cuda_device) {
        auto status = cudaSetDevice(cuda_device.id);
	if (status != cudaSuccess) {
	    LOG_ERROR("Error setting device: " << cuda_device.id << ", return status: " << status);
	}
	return status;
    }

    cudaError_t get_cuda_device(CudaDevice& cuda_device) {
	auto status = cudaGetDevice(&cuda_device.id);
	if (status != cudaSuccess) {
	    LOG_ERROR("Error getting device: " << cuda_device.id << ", return status: " << status);
	}

	cudaDeviceProp device_prop;
	auto ret = cudaGetDeviceProperties(&device_prop, cuda_device.id);
	if (ret != cudaSuccess) {
	    LOG_ERROR("Error getting device properties for device: " << cuda_device.id << ", return status: " << ret);
	}
	cuda_device.major = device_prop.major;
	cuda_device.minor = device_prop.minor;
	return ret;
    }

    std::string serialize_device(CudaDevice& cuda_device) {

	void *buffer = new char[sizeof(cuda_device)];
	void *ref_buf = buffer;

	memcpy(buffer, reinterpret_cast<int*>(&cuda_device.id), sizeof(int));
	//buffer += sizeof(int);
	buffer = static_cast<char*>(buffer) + sizeof(int);

	memcpy(buffer, reinterpret_cast<int*>(&cuda_device.major), sizeof(int));
	//buffer += sizeof(int);
	buffer = static_cast<char*>(buffer) + sizeof(int);

	memcpy(buffer, reinterpret_cast<int*>(&cuda_device.minor), sizeof(int));
	//buffer += sizeof(int);
	buffer = static_cast<char*>(buffer) + sizeof(int);

	return std::string((const char*)ref_buf, sizeof(int)*3);
    }

    CudaDevice deserialize_device(std::string device_info) {

        CudaDevice ret;
	char *buffer = new char[device_info.size() + 1];
	std::copy(device_info.begin(), device_info.end(), buffer);

	memcpy(&ret.id, reinterpret_cast<char*>(buffer), sizeof(int));
	buffer += sizeof(int);
	memcpy(&ret.major, reinterpret_cast<char*>(buffer), sizeof(int));
	buffer += sizeof(int);
	memcpy(&ret.minor, reinterpret_cast<char*>(buffer), sizeof(int));
	buffer += sizeof(int);

	return ret;
    }
}
}
}
