#pragma once
#include "core/util/logging/TRTorchLogger.h"
#include "core/util/Exception.h"

#define GET_MACRO(_1,_2,NAME,...) NAME

#define TRTORCH_LOG(l, sev, msg)               \
    do {                                       \
        std::stringstream ss{};                \
        ss << msg;                             \
        l.log(sev, ss.str());                  \
    } while (0)

#define LOG_GRAPH_GLOBAL(s)          TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kGRAPH, s)
#define LOG_DEBUG_GLOBAL(s)          TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kDEBUG, s)
#define LOG_INFO_GLOBAL(s)           TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kINFO, s)
#define LOG_WARNING_GLOBAL(s)        TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kWARNING, s)
#define LOG_ERROR_GLOBAL(s)          TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kERROR, s)
#define LOG_INTERNAL_ERROR_GLOBAL(s) TRTORCH_LOG(core::util::logging::get_logger(), core::util::logging::LogLevel::kINTERNAL_ERROR, s)

#define LOG_GRAPH_OWN(l,s)           TRTORCH_LOG(l, core::util::logging::LogLevel::kGRAPH, s)
#define LOG_DEBUG_OWN(l,s)           TRTORCH_LOG(l, core::util::logging::LogLevel::kDEBUG, s)
#define LOG_INFO_OWN(l,s)            TRTORCH_LOG(l, core::util::logging::LogLevel::kINFO, s)
#define LOG_WARNING_OWN(l,s)         TRTORCH_LOG(l, core::util::logging::LogLevel::kWARNING, s)
#define LOG_ERROR_OWN(l,s)           TRTORCH_LOG(l, core::util::logging::LogLevel::kERROR, s)
#define LOG_INTERNAL_ERROR_OWN(l,s)  TRTORCH_LOG(l, core::util::logging::LogLevel::kINTERNAL_ERROR, s)

#define LOG_GRAPH(...)          GET_MACRO(__VA_ARGS__, LOG_GRAPH_OWN, LOG_GRAPH_GLOBAL)(__VA_ARGS__)
#define LOG_DEBUG(...)          GET_MACRO(__VA_ARGS__, LOG_DEBUG_OWN, LOG_DEBUG_GLOBAL)(__VA_ARGS__)
#define LOG_INFO(...)           GET_MACRO(__VA_ARGS__, LOG_INFO_OWN, LOG_INFO_GLOBAL)(__VA_ARGS__)
#define LOG_WARNING(...)        GET_MACRO(__VA_ARGS__, LOG_WARNING_OWN, LOG_WARNING_GLOBAL)(__VA_ARGS__)
#define LOG_ERROR(...)          GET_MACRO(__VA_ARGS__, LOG_ERROR_OWN, LOG_ERROR_GLOBAL)(__VA_ARGS__)
#define LOG_INTERNAL_ERROR(...) GET_MACRO(__VA_ARGS__, LOG_INTERNAL_ERROR_OWN, LOG_INTERNAL_ERROR_GLOBAL)(__VA_ARGS__)

// ----------------------------------------------------------------------------
// Error reporting macros
// ----------------------------------------------------------------------------

#define TRTORCH_THROW_ERROR(msg) \
    std::stringstream ss{};      \
    ss << msg;                   \
    throw ::trtorch::Error(__FILE__, static_cast<uint32_t>(__LINE__), ss.str());

#define TRTORCH_ASSERT(cond, ...)                                                       \
    if (!(cond)) {                                                                      \
    TRTORCH_THROW_ERROR(#cond                                                           \
           << " ASSERT FAILED at "                                                      \
           << __FILE__ << ':'                                                           \
           << __LINE__                                                                  \
           << ", consider filing a bug: https://www.github.com/NVIDIA/TRTorch/issues\n" \
           << __VA_ARGS__);                                                             \
     }

#define TRTORCH_CHECK(cond, ...)                                            \
  if (!(cond)) {                                                            \
      TRTORCH_THROW_ERROR("Expected " << #cond                              \
                          << " to be true but got false\n" << __VA_ARGS__); \
  }


// suppress an unused variable.
#if defined(_MSC_VER) && !defined(__clang__)
#define TRTORCH_UNUSED __pragma(warning(suppress: 4100 4101))
#else
#define TRTORCH_UNUSED __attribute__((__unused__))
#endif //_MSC_VER
