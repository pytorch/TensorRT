#pragma once

#include <string>
#include <sstream>

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace util {

inline std::string node_info(const torch::jit::Node* n) {
    std::stringstream ss;
    ss << *n;
    std::string node_info = ss.str();
    node_info.erase(std::remove(node_info.begin(), node_info.end(), '\n'), node_info.end());
    return node_info;
}

inline std::string schema_info(const torch::jit::FunctionSchema* s) {
    std::stringstream ss;
    ss << *s;
    std::string schema_info = ss.str();
    schema_info.erase(std::remove(schema_info.begin(), schema_info.end(), '\n'), schema_info.end());
    return schema_info;
}

inline std::vector<int64_t> toVec(c10::IntArrayRef a) {
    std::vector<int64_t> arr;
    for (auto i : a) {
        arr.push_back(i);
    }
    return arr;
}

} // namespace util
} // namespace core
} // namespace trtorch
