#pragma once
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {
// Helper class which registers a plugin in torch_tensorrt namespace
template <typename T>
class PluginRegistrar {
 public:
  PluginRegistrar() {
    getPluginRegistry()->registerCreator(instance, "torch_tensorrt");
  }

 private:
  T instance{};
};

#define REGISTER_TORCHTRT_PLUGIN(name) \
  static PluginRegistrar<name> pluginRegistrar##name {}

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
