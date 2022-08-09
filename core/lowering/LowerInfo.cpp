#include <iostream>
#include <sstream>
#include <utility>

#include "core/lowering/lowering.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {

std::ostream& operator<<(std::ostream& os, const LowerInfo& l) {
  os << "Settings requested for Lowering:" << std::endl;
  os << "    torch_executed_modules: [" << std::endl;
  for (auto i : l.forced_fallback_modules) {
    os << "      " << i << std::endl;
  }
  os << "    ]";
  return os;
}

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
