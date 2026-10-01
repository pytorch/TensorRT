/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <filesystem>
#include <string>

#include "core/runtime/runtime.h"
#include "gtest/gtest.h"

namespace runtime = torch_tensorrt::core::runtime;

TEST(Runtime, GlobalProfilingConfiguration) {
  const auto original = runtime::get_global_profiling_config();
  const auto test_path = std::filesystem::temp_directory_path().string() + "/torchtrt_global_profiling_test";

  runtime::set_profile_path_prefix(test_path);
  runtime::set_profile_format("trex");
  runtime::set_profile_execution(true);

  const auto configured = runtime::get_global_profiling_config();
  EXPECT_TRUE(configured.enabled);
  EXPECT_EQ(configured.profile_format, "trex");
  EXPECT_EQ(configured.profile_path_prefix, test_path);
  EXPECT_GT(configured.generation, original.generation);
  EXPECT_EQ(runtime::get_global_profiling_generation(), configured.generation);
  EXPECT_TRUE(runtime::get_profile_execution());
  EXPECT_EQ(runtime::get_profile_format(), "trex");
  EXPECT_EQ(runtime::get_profile_path_prefix(), test_path);

  // Restore process-global state for any tests sharing this binary.
  runtime::set_profile_path_prefix(original.profile_path_prefix);
  runtime::set_profile_format(original.profile_format);
  runtime::set_profile_execution(original.enabled);
}
