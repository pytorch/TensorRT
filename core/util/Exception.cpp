#include "core/util/Exception.h"

#include <iostream>
#include <numeric>
#include <string>

namespace torch_tensorrt {

Error::Error(const std::string& new_msg, const void* caller) : msg_stack_{new_msg}, caller_(caller) {
  msg_ = msg();
}

Error::Error(const char* file, const uint32_t line, const std::string& msg, const void* caller)
    : Error(str("[Error thrown at ", file, ":", line, "] ", msg, "\n"), caller) {}

std::string Error::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

void Error::AppendMessage(const std::string& new_msg) {
  msg_stack_.push_back(new_msg);
  msg_ = msg();
}

std::string GetExceptionString(const std::exception& e) {
  return std::string("Exception: ") + e.what();
}

} // namespace torch_tensorrt
