#pragma once

#include <string>
#include "NvInfer.h"

namespace trtorch {
namespace core {
namespace util {
namespace logging {

enum class LogLevel : uint8_t {
  kINTERNAL_ERROR = (int)nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
  kERROR = (int)nvinfer1::ILogger::Severity::kERROR,
  kWARNING = (int)nvinfer1::ILogger::Severity::kWARNING,
  kINFO = (int)nvinfer1::ILogger::Severity::kINFO,
  kDEBUG = (int)nvinfer1::ILogger::Severity::kVERBOSE,
  kGRAPH
};

// Logger for TensorRT info/warning/errors
class TRTorchLogger : public nvinfer1::ILogger {
 public:
  TRTorchLogger(std::string prefix = "[TRTorch] - ", Severity severity = Severity::kWARNING, bool color = true);
  TRTorchLogger(std::string prefix = "[TRTorch] - ", LogLevel lvl = LogLevel::kWARNING, bool color = true);
  void log(Severity severity, const char* msg) noexcept override;
  void log(LogLevel lvl, std::string msg);
  void set_logging_prefix(std::string prefix);
  void set_reportable_severity(Severity severity);
  void set_reportable_log_level(LogLevel severity);
  void set_is_colored_output_on(bool colored_output_on);
  std::string get_logging_prefix();
  Severity get_reportable_severity();
  LogLevel get_reportable_log_level();
  bool get_is_colored_output_on();

 private:
  std::string prefix_;
  LogLevel reportable_severity_;
  bool color_;
};

TRTorchLogger& get_logger();

} // namespace logging
} // namespace util
} // namespace core
} // namespace trtorch
