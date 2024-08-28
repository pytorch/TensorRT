#pragma once
#include <string>
#include <unordered_map>

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct Platform {
  typedef enum {
    kLINUX_X86_64 = 0,
    kLINUX_AARCH64,
    kWIN_X86_64,
    kUNKNOWN,
  } PlatformEnum;

  PlatformEnum _platform = Platform::kUNKNOWN;

  Platform();
  Platform(PlatformEnum val);
  Platform(const std::string& platform_str);
  std::string serialize() const;
  Platform& operator=(const Platform& other);

  friend std::ostream& operator<<(std::ostream& os, const Platform& device);
  friend bool operator==(const Platform& lhs, const Platform& rhs);
};

const std::unordered_map<Platform::PlatformEnum, std::string>& get_platform_name_map();
Platform get_current_platform();
bool is_supported_on_current_platform(Platform target);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
