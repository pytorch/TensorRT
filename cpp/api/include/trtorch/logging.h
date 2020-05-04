#pragma once

#include <string>
#include "trtorch/macros.h"

namespace trtorch {
namespace logging {
/**
 * Emum for setting message severity
 */
enum Level {
    kINTERNAL_ERROR, // Only print messages for internal errors
    kERROR,          // Print all internal errors and errors (default)
    kWARNING,        // Print warnings and errors
    kINFO,           // Print all info, warnings and errors
    kDEBUG,          // Print all debug info, info, warnings and errors
    kGRAPH,          // Print everything including the intermediate graphs of the lowering phase
};

// Are these ones necessary for the user?
TRTORCH_API std::string get_logging_prefix();
TRTORCH_API void set_logging_prefix(std::string prefix);

/**
 * @brief Sets the level that logging information needs to be to be added to the log
 *
 * @param lvl: trtorch::logging::Level - Level that messages need to be at or above to be added to the log
 */
TRTORCH_API void set_reportable_log_level(Level lvl);

/**
 * @brief Sets if logging prefix will be colored (helpful when debugging but not always supported by terminal)
 *
 * @param colored_output_on: bool - If the output will be colored or not
 */
TRTORCH_API void set_is_colored_output_on(bool colored_output_on);

/**
 * @brief Get the current reportable log level
 */
TRTORCH_API Level get_reportable_log_level();

/**
 * @brief Is colored output enabled?
 */
TRTORCH_API bool get_is_colored_output_on();

/**
 * @brief Adds a message to the global log
 *
 * @param lvl: trtorch::logging::Level - Severity of the message
 * @param msg: std::string - Message to be logged
 */
// Dont know if we want this?
TRTORCH_API void log(Level lvl, std::string msg);
} // namespace logging
} // namespace trtorch
