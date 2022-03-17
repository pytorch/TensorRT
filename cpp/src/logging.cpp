#include "torch_tensorrt/logging.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace logging {

std::string get_logging_prefix() {
  return torchtrt::core::util::logging::get_logger().get_logging_prefix();
}

void set_logging_prefix(std::string prefix) {
  torchtrt::core::util::logging::get_logger().set_logging_prefix(prefix);
}

void set_reportable_log_level(Level lvl) {
  torchtrt::core::util::logging::LogLevel log_lvl;
  switch (lvl) {
    case Level::kINTERNAL_ERROR:
      log_lvl = torchtrt::core::util::logging::LogLevel::kINTERNAL_ERROR;
      break;
    case Level::kERROR:
      log_lvl = torchtrt::core::util::logging::LogLevel::kERROR;
      break;
    case Level::kWARNING:
      log_lvl = torchtrt::core::util::logging::LogLevel::kWARNING;
      break;
    case Level::kINFO:
      log_lvl = torchtrt::core::util::logging::LogLevel::kINFO;
      break;
    case Level::kGRAPH:
      log_lvl = torchtrt::core::util::logging::LogLevel::kGRAPH;
      break;
    case Level::kDEBUG:
    default:
      log_lvl = torchtrt::core::util::logging::LogLevel::kDEBUG;
  }
  torchtrt::core::util::logging::get_logger().set_reportable_log_level(log_lvl);
}

void set_is_colored_output_on(bool colored_output_on) {
  torchtrt::core::util::logging::get_logger().set_is_colored_output_on(colored_output_on);
}

Level get_reportable_log_level() {
  switch (torchtrt::core::util::logging::get_logger().get_reportable_log_level()) {
    case torchtrt::core::util::logging::LogLevel::kINTERNAL_ERROR:
      return Level::kINTERNAL_ERROR;
    case torchtrt::core::util::logging::LogLevel::kERROR:
      return Level::kERROR;
    case torchtrt::core::util::logging::LogLevel::kWARNING:
      return Level::kWARNING;
    case torchtrt::core::util::logging::LogLevel::kINFO:
      return Level::kINFO;
    case torchtrt::core::util::logging::LogLevel::kGRAPH:
      return Level::kGRAPH;
    case torchtrt::core::util::logging::LogLevel::kDEBUG:
    default:
      return Level::kDEBUG;
  }
}

bool get_is_colored_output_on() {
  return torchtrt::core::util::logging::get_logger().get_is_colored_output_on();
}

void log(Level lvl, std::string msg) {
  torchtrt::core::util::logging::get_logger().log((torchtrt::core::util::logging::LogLevel)(lvl), msg);
}
} // namespace logging
} // namespace torch_tensorrt
