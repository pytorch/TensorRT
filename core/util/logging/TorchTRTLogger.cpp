#include "core/util/logging/TorchTRTLogger.h"

#include <iostream>
#include <string>

#define TERM_NORMAL "\033[0m";
#define TERM_RED "\033[0;31m";
#define TERM_YELLOW "\033[0;33m";
#define TERM_GREEN "\033[0;32m";
#define TERM_MAGENTA "\033[1;35m";

namespace torch_tensorrt {
namespace core {
namespace trt = nvinfer1;

namespace util {
namespace logging {

TorchTRTLogger::TorchTRTLogger(std::string prefix, Severity severity, bool color)
    : prefix_(prefix), reportable_severity_((LogLevel)severity), color_(color) {}

TorchTRTLogger::TorchTRTLogger(std::string prefix, LogLevel lvl, bool color)
    : prefix_(prefix), reportable_severity_(lvl), color_(color) {}

void TorchTRTLogger::log(LogLevel lvl, std::string msg) {
  // suppress messages with severity enum value greater than the reportable
  if (lvl > reportable_severity_) {
    return;
  }

  if (color_) {
    switch (lvl) {
      case LogLevel::kINTERNAL_ERROR:
        std::cerr << TERM_RED;
        break;
      case LogLevel::kERROR:
        std::cerr << TERM_RED;
        break;
      case LogLevel::kWARNING:
        std::cerr << TERM_YELLOW;
        break;
      case LogLevel::kINFO:
        std::cerr << TERM_GREEN;
        break;
      case LogLevel::kDEBUG:
        std::cerr << TERM_MAGENTA;
        break;
      case LogLevel::kGRAPH:
        std::cerr << TERM_NORMAL;
        break;
      default:
        break;
    }
  }

  switch (lvl) {
    case LogLevel::kINTERNAL_ERROR:
      std::cerr << "INTERNAL_ERROR: ";
      break;
    case LogLevel::kERROR:
      std::cerr << "ERROR: ";
      break;
    case LogLevel::kWARNING:
      std::cerr << "WARNING: ";
      break;
    case LogLevel::kINFO:
      std::cerr << "INFO: ";
      break;
    case LogLevel::kDEBUG:
      std::cerr << "DEBUG: ";
      break;
    case LogLevel::kGRAPH:
      std::cerr << "GRAPH: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
      break;
  }

  if (color_) {
    std::cerr << TERM_NORMAL;
  }

  std::cerr << prefix_ << msg << std::endl;
}

void TorchTRTLogger::log(Severity severity, const char* msg) noexcept {
  LogLevel lvl = (LogLevel)severity;
  log(lvl, std::string(msg));
}

void TorchTRTLogger::set_logging_prefix(std::string prefix) {
  prefix_ = prefix;
}

void TorchTRTLogger::set_reportable_severity(Severity severity) {
  reportable_severity_ = (LogLevel)severity;
}

void TorchTRTLogger::set_reportable_log_level(LogLevel lvl) {
  reportable_severity_ = lvl;
}

void TorchTRTLogger::set_is_colored_output_on(bool colored_output_on) {
  color_ = colored_output_on;
}

std::string TorchTRTLogger::get_logging_prefix() {
  return prefix_;
}

nvinfer1::ILogger::Severity TorchTRTLogger::get_reportable_severity() {
  return (Severity)reportable_severity_;
}

LogLevel TorchTRTLogger::get_reportable_log_level() {
  return reportable_severity_;
}

bool TorchTRTLogger::get_is_colored_output_on() {
  return color_;
}

namespace {

TorchTRTLogger& get_global_logger() {
#ifndef NDEBUG
  static TorchTRTLogger global_logger("[Torch-TensorRT - Debug Build] - ", LogLevel::kDEBUG, true);
#else
  static TorchTRTLogger global_logger("[Torch-TensorRT] - ", LogLevel::kWARNING, false);
#endif
  return global_logger;
}

} // namespace

TorchTRTLogger& get_logger() {
  return get_global_logger();
}

} // namespace logging
} // namespace util
} // namespace core
} // namespace torch_tensorrt
