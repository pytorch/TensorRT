#include <string>
#include "core/compiler.h"
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"

TEST(API, TRTorchSetDeviceTest) {
  // Check number of CUDA capable device on the target
  int device_count = -1;
  assert(cudaGetDeviceCount(&device_count) == cudaSuccess);
  assert(device_count != 0);

  int gpu_id = device_count - 1;
  trtorch::core::set_device(gpu_id);

  // Verify if the device ID is set correctly
  int device = -1;
  assert(cudaGetDevice(&device) == cudaSuccess);

  ASSERT_TRUE(device == gpu_id);
}
