#include "core/runtime/Platform.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {
const std::unordered_map<std::string, Platform::PlatformEnum>& get_name_to_platform_map() {
  static const std::unordered_map<std::string, Platform::PlatformEnum> name_to_platform_map = {
      {"linux_aarch64", Platform::PlatformEnum::kLINUX_AARCH64},
      {"linux_x86_64", Platform::PlatformEnum::kLINUX_X86_64},
      {"windows_x86_64", Platform::PlatformEnum::kWIN_X86_64},
      {"unknown", Platform::PlatformEnum::kUNKNOWN},
  };
  return name_to_platform_map;
}

const std::unordered_map<Platform::PlatformEnum, std::string>& _get_platform_name_map() {
  static const std::unordered_map<Platform::PlatformEnum, std::string> platform_name_map = {
      {Platform::PlatformEnum::kLINUX_AARCH64, "linux_aarch64"},
      {Platform::PlatformEnum::kLINUX_X86_64, "linux_x86_64"},
      {Platform::PlatformEnum::kWIN_X86_64, "windows_x86_64"},
      {Platform::PlatformEnum::kUNKNOWN, "unknown"}};
  return platform_name_map;
}
} // namespace

const std::unordered_map<Platform::PlatformEnum, std::string>& get_platform_name_map() {
  return _get_platform_name_map();
}

Platform::Platform() : _platform{Platform::PlatformEnum::kUNKNOWN} {}

Platform::Platform(Platform::PlatformEnum val) : _platform{val} {}

Platform::Platform(const std::string& platform_str) {
  auto name_map = get_name_to_platform_map();
  auto it = name_map.find(platform_str);
  if (it != name_map.end()) {
    _platform = it->second;
  } else {
    LOG_WARNING("Unknown platform " << platform_str);
    _platform = Platform::PlatformEnum::kUNKNOWN;
  }
}

std::string Platform::serialize() const {
  auto name_map = get_platform_name_map();
  auto it = name_map.find(_platform);
  if (it != name_map.end()) {
    return it->second;
  } else {
    LOG_WARNING("Attempted to serialized unknown platform tag");
    return std::string("unknown");
  }
}

Platform& Platform::operator=(const Platform& other) {
  _platform = other._platform;
  return (*this);
}

bool operator==(const Platform& lhs, const Platform& rhs) {
  return lhs._platform == rhs._platform;
}

std::ostream& operator<<(std::ostream& os, const Platform& platform) {
  os << platform.serialize();
  return os;
}

Platform get_current_platform() {
#if defined(__linux__) || defined(__gnu_linux__)
#if defined(__aarch64__)
  return Platform(Platform::PlatformEnum::kLINUX_AARCH64);
#elif defined(__amd64__) || defined(__x86_64__)
  return Platform(Platform::PlatformEnum::kLINUX_X86_64);
#else
  return Platform(Platform::PlatformEnum::kUNKNOWN);
#endif
#elif defined(_WIN32) || defined(_WIN64)
#if defined(_M_AMD64) || defined(_M_X64)
  return Platform(Platform::PlatformEnum::kWIN_X86_64);
#else
  return Platform(Platform::PlatformEnum::kUNKNOWN);
#endif
#else
  return Platform(Platform::PlatformEnum::kUNKNOWN);
#endif
}

bool is_supported_on_current_platform(Platform target) {
  // Space for more complicated platform support calculations later
  return target == get_current_platform();
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
