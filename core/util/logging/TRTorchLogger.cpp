#include "core/util/logging/TRTorchLogger.h"

#include <iostream>
#include <string>

#define TERM_NORMAL  "\033[0m";
#define TERM_RED     "\033[0;31m";
#define TERM_YELLOW  "\033[0;33m";
#define TERM_GREEN   "\033[0;32m";
#define TERM_MAGENTA "\033[1;35m";

namespace trtorch {
namespace core {
namespace trt = nvinfer1;

namespace util {
namespace logging {
    
TRTorchLogger::TRTorchLogger(std::string prefix, Severity severity, bool color)
    : prefix_(prefix), reportable_severity_(severity), color_(color) {}

TRTorchLogger::TRTorchLogger(std::string prefix, LogLevel lvl, bool color)
    : prefix_(prefix), reportable_severity_((Severity) lvl), color_(color) {}

void TRTorchLogger::log(LogLevel lvl, std::string msg) {
    Severity severity = (Severity) lvl;
    log(severity, msg.c_str());
}

void TRTorchLogger::log(Severity severity, const char* msg) {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity_) {
        return;
    }
    
    if (color_) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR: std::cerr << TERM_RED; break;
        case Severity::kERROR: std::cerr << TERM_RED; break;
        case Severity::kWARNING: std::cerr << TERM_YELLOW; break;
        case Severity::kINFO: std::cerr << TERM_GREEN; break;
        case Severity::kVERBOSE: std::cerr << TERM_MAGENTA; break;
        default: break;
        }       
    }
    
    switch (severity) {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
    case Severity::kERROR: std::cerr << "ERROR: "; break;
    case Severity::kWARNING: std::cerr << "WARNING: "; break;
    case Severity::kINFO: std::cerr << "INFO: "; break;
    case Severity::kVERBOSE: std::cerr << "DEBUG: "; break;
    default: std::cerr << "UNKNOWN: "; break;
    }
    
    if (color_) {
        std::cerr << TERM_NORMAL;
    }
    
    std::cerr << prefix_ << msg << std::endl;
}

void TRTorchLogger::set_logging_prefix(std::string prefix) {
    prefix_ = prefix;
}

void TRTorchLogger::set_reportable_severity(Severity severity) {
    reportable_severity_ = severity;
}

void TRTorchLogger::set_reportable_log_level(LogLevel lvl) {
    reportable_severity_ = (Severity) lvl;
}

void TRTorchLogger::set_is_colored_output_on(bool colored_output_on) {
    color_ = colored_output_on;
}

std::string TRTorchLogger::get_logging_prefix() {
    return prefix_;
}

nvinfer1::ILogger::Severity TRTorchLogger::get_reportable_severity() {
    return reportable_severity_;
}

LogLevel TRTorchLogger::get_reportable_log_level() {
    return (LogLevel) reportable_severity_;
}

bool TRTorchLogger::get_is_colored_output_on() {
    return color_;
}

    
namespace {

TRTorchLogger& get_global_logger() {
    #ifndef NDEBUG
    static TRTorchLogger global_logger("[TRTorch - Debug Build] - ",
                                       LogLevel::kDEBUG,
                                       true);
    #else
    static TRTorchLogger global_logger("[TRTorch] - ",
                                       LogLevel::kERROR,
                                       false);
    #endif 
    return global_logger;
}

} // namespace

TRTorchLogger& get_logger() {
    return get_global_logger();
}

} // namespace logging
} // namespace util
} // namespace core
} // namespace trtorch
