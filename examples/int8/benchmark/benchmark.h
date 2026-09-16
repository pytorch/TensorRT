/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

void print_avg_std_dev(std::string type, std::vector<float>& runtimes, uint64_t batch_size);
std::vector<float> benchmark_module(torch::jit::script::Module& mod, std::vector<int64_t> shape);
