#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <vector>

// Simplified version of the c10 Exception infrastructure
// https://github.com/pytorch/pytorch/blob/master/c10/util/Exception.h

// Hopefully makes it easer to merge codebases later and still gives better
// errors

namespace torch_tensorrt {

class Error : public std::exception {
  std::vector<std::string> msg_stack_;
  std::string msg_;
  const void* caller_;

 public:
  Error(const std::string& msg, const void* caller = nullptr);
  Error(const char* file, const uint32_t line, const std::string& msg, const void* caller = nullptr);

  void AppendMessage(const std::string& msg);

  std::string msg() const;

  const std::vector<std::string>& msg_stack() const {
    return msg_stack_;
  }

  const char* what() const noexcept override {
    return msg_.c_str();
  }

  const void* caller() const noexcept {
    return caller_;
  }
};

std::string GetExceptionString(const std::exception& e);

namespace detail {
inline std::string if_empty_then(std::string x, std::string y) {
  if (x.empty()) {
    return y;
  } else {
    return x;
  }
}

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  ss << t;
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
inline std::string _str_wrapper(const Args&... args) {
  std::ostringstream ss;
  _str(ss, args...);
  return ss.str();
}
} // namespace detail

template <typename... Args>
inline std::string str(const Args&... args) {
  return detail::_str_wrapper<typename detail::CanonicalizeStrTypes<Args>::type...>(args...);
}

template <>
inline std::string str(const std::string& str) {
  return str;
}

inline std::string str(const char* c_str) {
  return c_str;
}
} // namespace torch_tensorrt
