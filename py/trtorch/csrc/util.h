#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <functional>
#include <iostream>
#include <string>
#include "core/util/logging/TRTorchLogger.h"

namespace trtorch {
namespace pyapi {
namespace util {

namespace py = pybind11;

// Method for calling the python function and returning the value (returned from python) used in cpp trampoline
// classes. Prints an error if no such method is overriden in python.
// T* must NOT be a trampoline class!
template <typename T>
py::function getOverload(const T* self, const std::string& overloadName) {
  py::function overload = py::get_override(self, overloadName.c_str());
  if (!overload) {
    std::string msg{"Method: " + overloadName +
                    " was not overriden. Please provide an implementation for this method."};
    // std::cerr << "Method: " << overloadName << " was not overriden. Please provide an implementation for this
    // method.";
    core::util::logging::get_logger().log(core::util::logging::LogLevel::kERROR, msg);
  }
  return overload;
}

} // namespace util
} // namespace pyapi
} // namespace trtorch
