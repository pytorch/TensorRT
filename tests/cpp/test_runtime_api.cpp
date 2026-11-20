/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime.h>
#include "core/runtime/runtime.h"
#include "core/util/logging/TorchTRTLogger.h"
#include "gtest/gtest.h"
#include "torch_tensorrt/torch_tensorrt.h"

namespace {

class RuntimeAPI : public testing::Test {
 protected:
  void TearDown() override {
    torch_tensorrt::logging::set_reportable_log_level(level_);
    torch_tensorrt::logging::set_logging_prefix(prefix_);
    torch_tensorrt::logging::set_is_colored_output_on(color_);
    torch_tensorrt::core::runtime::set_multi_device_safe_mode(safe_mode_);
    torch_tensorrt::core::runtime::set_cudagraphs_mode(cudagraphs_);
  }

 private:
  torch_tensorrt::logging::Level level_ = torch_tensorrt::logging::get_reportable_log_level();
  std::string prefix_ = torch_tensorrt::logging::get_logging_prefix();
  bool color_ = torch_tensorrt::logging::get_is_colored_output_on();
  bool safe_mode_ = torch_tensorrt::core::runtime::get_multi_device_safe_mode();
  torch_tensorrt::core::runtime::CudaGraphsMode cudagraphs_ = torch_tensorrt::core::runtime::get_cudagraphs_mode();
};

TEST_F(RuntimeAPI, LoggingControlsTensorRTLogger) {
  namespace logging = torch_tensorrt::logging;
  auto& logger = torch_tensorrt::core::util::logging::get_logger();
  logging::set_logging_prefix("[AOTI application] ");
  logging::set_is_colored_output_on(false);
  logging::set_reportable_log_level(logging::Level::kWARNING);

  testing::internal::CaptureStderr();
  logger.log(nvinfer1::ILogger::Severity::kINFO, "suppressed");
  logger.log(nvinfer1::ILogger::Severity::kWARNING, "runtime warning");
  logging::log(logging::Level::kERROR, "application error");
  auto output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(output.find("suppressed"), std::string::npos);
  EXPECT_NE(output.find("[AOTI application] runtime warning"), std::string::npos);
  EXPECT_NE(output.find("[AOTI application] application error"), std::string::npos);

  for (auto level :
       {logging::kINTERNAL_ERROR,
        logging::kERROR,
        logging::kWARNING,
        logging::kINFO,
        logging::kDEBUG,
        logging::kGRAPH}) {
    logging::set_reportable_log_level(level);
    EXPECT_EQ(logging::get_reportable_log_level(), level);
    EXPECT_EQ(static_cast<int>(logger.get_reportable_log_level()), static_cast<int>(level));
  }
}

TEST_F(RuntimeAPI, BuildInfoRemainsAvailable) {
  auto info = torch_tensorrt::get_build_info();
  EXPECT_NE(info.find(TORCH_TENSORRT_VERSION), std::string::npos);
  EXPECT_NE(info.find("Using TensorRT Version:"), std::string::npos);
  testing::internal::CaptureStdout();
  torch_tensorrt::dump_build_info();
  EXPECT_EQ(testing::internal::GetCapturedStdout(), info + "\n");
}

TEST_F(RuntimeAPI, ExecutionControlsRemainAvailable) {
  namespace runtime = torch_tensorrt::core::runtime;
  runtime::set_multi_device_safe_mode(true);
  EXPECT_TRUE(runtime::get_multi_device_safe_mode());
  runtime::set_multi_device_safe_mode(false);
  EXPECT_FALSE(runtime::get_multi_device_safe_mode());
  runtime::set_cudagraphs_mode(runtime::SUBGRAPH_CUDAGRAPHS);
  EXPECT_EQ(runtime::get_cudagraphs_mode(), runtime::SUBGRAPH_CUDAGRAPHS);
  runtime::set_cudagraphs_mode(runtime::STANDARD);
  EXPECT_EQ(runtime::get_cudagraphs_mode(), runtime::STANDARD);
}

TEST_F(RuntimeAPI, DeviceSelectionRemainsAvailable) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "requires a CUDA device";
  }
  int device = -1;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  torch_tensorrt::set_device(device);
  int selected = -1;
  ASSERT_EQ(cudaGetDevice(&selected), cudaSuccess);
  EXPECT_EQ(selected, device);
}

} // namespace
