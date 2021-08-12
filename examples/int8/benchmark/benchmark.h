#pragma once

void print_avg_std_dev(std::string type, std::vector<float>& runtimes, uint64_t batch_size);
std::vector<float> benchmark_module(torch::jit::script::Module& mod, std::vector<int64_t> shape);
