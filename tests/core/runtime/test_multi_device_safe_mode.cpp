#include "core/runtime/runtime.h"
#include "gtest/gtest.h"

TEST(Runtime, MultiDeviceSafeMode) {
  ASSERT_TRUE(!torch_tensorrt::core::runtime::get_multi_device_safe_mode());
  torch_tensorrt::core::runtime::set_multi_device_safe_mode(true);
  ASSERT_TRUE(torch_tensorrt::core::runtime::get_multi_device_safe_mode());
}
