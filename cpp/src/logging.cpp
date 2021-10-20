#include "torch_tensorrt/logging.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace logging {

std::string get_logging_prefix() {
  return trtorch::core::util::logging::get_logger().get_logging_prefix();
}

void set_logging_prefix(std::string prefix) {
  trtorch::core::util::logging::get_logger().set_logging_prefix(prefix);
}

void set_reportable_log_level(Level lvl) {
  trtorch::core::util::logging::LogLevel log_lvl;
  switch (lvl) {
    case Level::kINTERNAL_ERROR:
      log_lvl = trtorch::core::util::logging::LogLevel::kINTERNAL_ERROR;
      break;
    case Level::kERROR:
      log_lvl = trtorch::core::util::logging::LogLevel::kERROR;
      break;
    case Level::kWARNING:
      log_lvl = trtorch::core::util::logging::LogLevel::kWARNING;
      break;
    case Level::kINFO:
      log_lvl = trtorch::core::util::logging::LogLevel::kINFO;
      break;
    case Level::kGRAPH:
      log_lvl = trtorch::core::util::logging::LogLevel::kGRAPH;
      break;
    case Level::kDEBUG:
    default:
      log_lvl = trtorch::core::util::logging::LogLevel::kDEBUG;
  }
  trtorch::core::util::logging::get_logger().set_reportable_log_level(log_lvl);
}

void set_is_colored_output_on(bool colored_output_on) {
  trtorch::core::util::logging::get_logger().set_is_colored_output_on(colored_output_on);
}

Level get_reportable_log_level() {
  switch (trtorch::core::util::logging::get_logger().get_reportable_log_level()) {
    case trtorch::core::util::logging::LogLevel::kINTERNAL_ERROR:
      return Level::kINTERNAL_ERROR;
    case trtorch::core::util::logging::LogLevel::kERROR:
      return Level::kERROR;
    case trtorch::core::util::logging::LogLevel::kWARNING:
      return Level::kWARNING;
    case trtorch::core::util::logging::LogLevel::kINFO:
      return Level::kINFO;
    case trtorch::core::util::logging::LogLevel::kGRAPH:
      return Level::kGRAPH;
    case trtorch::core::util::logging::LogLevel::kDEBUG:
    default:
      return Level::kDEBUG;
  }
}

bool get_is_colored_output_on() {
  return trtorch::core::util::logging::get_logger().get_is_colored_output_on();
}

void log(Level lvl, std::string msg) {
  trtorch::core::util::logging::get_logger().log((trtorch::core::util::logging::LogLevel)(lvl), msg);
}
} // namespace logging
} // namespace torchtrt
