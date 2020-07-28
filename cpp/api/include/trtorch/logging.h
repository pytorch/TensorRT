/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This library is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <string>
#include "trtorch/macros.h"

namespace trtorch {
namespace logging {
/**
 * Emum for setting message severity
 */
enum Level {
    /// Only print messages for internal errors
    kINTERNAL_ERROR,
    /// Print all internal errors and errors (default)
    kERROR,
    /// Print warnings and errors
    kWARNING,
    /// Print all info, warnings and errors
    kINFO,
    /// Print all debug info, info, warnings and errors
    kDEBUG,
    /// Print everything including the intermediate graphs of the lowering phase
    kGRAPH,
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
 *
 * @return TRTORCH_API get_reportable_log_level
 */
TRTORCH_API Level get_reportable_log_level();

/**
 * @brief Is colored output enabled?
 *
 * @return TRTORCH_API get_is_colored_output_on
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
