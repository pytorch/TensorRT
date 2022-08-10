#pragma once
#include "core/util/Exception.h"
#include "core/util/logging/TorchTRTLogger.h"

#define GET_MACRO(_1, _2, NAME, ...) NAME

// DLA Memory related macros
#define DLA_SRAM_SIZE 1048576
#define DLA_LOCAL_DRAM_SIZE 1073741824
#define DLA_GLOBAL_DRAM_SIZE 536870912

#define TORCHTRT_LOG(l, sev, msg) \
  do {                            \
    std::stringstream ss{};       \
    ss << msg;                    \
    l.log(sev, ss.str());         \
  } while (0)

#define LOG_GRAPH_GLOBAL(s) \
  TORCHTRT_LOG(             \
      torch_tensorrt::core::util::logging::get_logger(), torch_tensorrt::core::util::logging::LogLevel::kGRAPH, s)
#define LOG_DEBUG_GLOBAL(s) \
  TORCHTRT_LOG(             \
      torch_tensorrt::core::util::logging::get_logger(), torch_tensorrt::core::util::logging::LogLevel::kDEBUG, s)
#define LOG_INFO_GLOBAL(s) \
  TORCHTRT_LOG(            \
      torch_tensorrt::core::util::logging::get_logger(), torch_tensorrt::core::util::logging::LogLevel::kINFO, s)
#define LOG_WARNING_GLOBAL(s) \
  TORCHTRT_LOG(               \
      torch_tensorrt::core::util::logging::get_logger(), torch_tensorrt::core::util::logging::LogLevel::kWARNING, s)
#define LOG_ERROR_GLOBAL(s) \
  TORCHTRT_LOG(             \
      torch_tensorrt::core::util::logging::get_logger(), torch_tensorrt::core::util::logging::LogLevel::kERROR, s)
#define LOG_INTERNAL_ERROR_GLOBAL(s)                                  \
  TORCHTRT_LOG(                                                       \
      torch_tensorrt::core::util::logging::get_logger(),              \
      torch_tensorrt::core::util::logging::LogLevel::kINTERNAL_ERROR, \
      s)

#define LOG_GRAPH_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kGRAPH, s)
#define LOG_DEBUG_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kDEBUG, s)
#define LOG_INFO_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kINFO, s)
#define LOG_WARNING_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kWARNING, s)
#define LOG_ERROR_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kERROR, s)
#define LOG_INTERNAL_ERROR_OWN(l, s) TORCHTRT_LOG(l, torch_tensorrt::core::util::logging::LogLevel::kINTERNAL_ERROR, s)

#ifdef _MSC_VER

#define EXPAND(x) x

#define LOG_GRAPH(...) EXPAND(GET_MACRO(__VA_ARGS__, LOG_GRAPH_OWN, LOG_GRAPH_GLOBAL)(__VA_ARGS__))
#define LOG_DEBUG(...) EXPAND(GET_MACRO(__VA_ARGS__, LOG_DEBUG_OWN, LOG_DEBUG_GLOBAL)(__VA_ARGS__))
#define LOG_INFO(...) EXPAND(GET_MACRO(__VA_ARGS__, LOG_INFO_OWN, LOG_INFO_GLOBAL)(__VA_ARGS__))
#define LOG_WARNING(...) EXPAND(GET_MACRO(__VA_ARGS__, LOG_WARNING_OWN, LOG_WARNING_GLOBAL)(__VA_ARGS__))
#define LOG_ERROR(...) EXPAND(GET_MACRO(__VA_ARGS__, LOG_ERROR_OWN, LOG_ERROR_GLOBAL)(__VA_ARGS__))
#define LOG_INTERNAL_ERROR(...) \
  EXPAND(GET_MACRO(__VA_ARGS__, LOG_INTERNAL_ERROR_OWN, LOG_INTERNAL_ERROR_GLOBAL)(__VA_ARGS__))

#else

#define LOG_GRAPH(...) GET_MACRO(__VA_ARGS__, LOG_GRAPH_OWN, LOG_GRAPH_GLOBAL)(__VA_ARGS__)
#define LOG_DEBUG(...) GET_MACRO(__VA_ARGS__, LOG_DEBUG_OWN, LOG_DEBUG_GLOBAL)(__VA_ARGS__)
#define LOG_INFO(...) GET_MACRO(__VA_ARGS__, LOG_INFO_OWN, LOG_INFO_GLOBAL)(__VA_ARGS__)
#define LOG_WARNING(...) GET_MACRO(__VA_ARGS__, LOG_WARNING_OWN, LOG_WARNING_GLOBAL)(__VA_ARGS__)
#define LOG_ERROR(...) GET_MACRO(__VA_ARGS__, LOG_ERROR_OWN, LOG_ERROR_GLOBAL)(__VA_ARGS__)
#define LOG_INTERNAL_ERROR(...)                                             \
  GET_MACRO(__VA_ARGS__, LOG_INTERNAL_ERROR_OWN, LOG_INTERNAL_ERROR_GLOBAL) \
  (__VA_ARGS__)

#endif
// ----------------------------------------------------------------------------
// Error reporting macros
// ----------------------------------------------------------------------------

#define TORCHTRT_THROW_ERROR(msg) \
  std::stringstream ss{};         \
  ss << msg;                      \
  throw ::torch_tensorrt::Error(__FILE__, static_cast<uint32_t>(__LINE__), ss.str());

#define TORCHTRT_ASSERT(cond, ...)                                                                \
  if (!(cond)) {                                                                                  \
    TORCHTRT_THROW_ERROR(                                                                         \
        #cond << " ASSERT FAILED at " << __FILE__ << ':' << __LINE__                              \
              << ", consider filing a bug: https://www.github.com/NVIDIA/Torch-TensorRT/issues\n" \
              << __VA_ARGS__);                                                                    \
  }

#define TORCHTRT_CHECK(cond, ...)                                                               \
  if (!(cond)) {                                                                                \
    TORCHTRT_THROW_ERROR("Expected " << #cond << " to be true but got false\n" << __VA_ARGS__); \
  }

// suppress an unused variable.
#if defined(_MSC_VER) && !defined(__clang__)
#define TORCHTRT_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define TORCHTRT_UNUSED __attribute__((__unused__))
#endif //_MSC_VER
