#include "trtorch/logging.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace logging {

std::string get_logging_prefix() {
  return core::util::logging::get_logger().get_logging_prefix();
}

void set_logging_prefix(std::string prefix) {
  core::util::logging::get_logger().set_logging_prefix(prefix);
}

void set_reportable_log_level(Level lvl) {
  core::util::logging::LogLevel log_lvl;
  switch (lvl) {
    case Level::kINTERNAL_ERROR:
      log_lvl = core::util::logging::LogLevel::kINTERNAL_ERROR;
      break;
    case Level::kERROR:
      log_lvl = core::util::logging::LogLevel::kERROR;
      break;
    case Level::kWARNING:
      log_lvl = core::util::logging::LogLevel::kWARNING;
      break;
    case Level::kINFO:
      log_lvl = core::util::logging::LogLevel::kINFO;
      break;
    case Level::kGRAPH:
      log_lvl = core::util::logging::LogLevel::kGRAPH;
      break;
    case Level::kDEBUG:
    default:
      log_lvl = core::util::logging::LogLevel::kDEBUG;
  }
  core::util::logging::get_logger().set_reportable_log_level(log_lvl);
}

void set_is_colored_output_on(bool colored_output_on) {
  core::util::logging::get_logger().set_is_colored_output_on(colored_output_on);
}

Level get_reportable_log_level() {
  switch (core::util::logging::get_logger().get_reportable_log_level()) {
    case core::util::logging::LogLevel::kINTERNAL_ERROR:
      return Level::kINTERNAL_ERROR;
    case core::util::logging::LogLevel::kERROR:
      return Level::kERROR;
    case core::util::logging::LogLevel::kWARNING:
      return Level::kWARNING;
    case core::util::logging::LogLevel::kINFO:
      return Level::kINFO;
    case core::util::logging::LogLevel::kGRAPH:
      return Level::kGRAPH;
    case core::util::logging::LogLevel::kDEBUG:
    default:
      return Level::kDEBUG;
  }
}

bool get_is_colored_output_on() {
  return core::util::logging::get_logger().get_is_colored_output_on();
}

void log(Level lvl, std::string msg) {
  core::util::logging::get_logger().log((core::util::logging::LogLevel)(lvl), msg);
}
} // namespace logging
} // namespace trtorch
