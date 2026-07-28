#pragma once
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {
// Helper class which registers a V2 plugin in torch_tensorrt namespace
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

// Helper class which registers a V3 plugin in torch_tensorrt namespace
template <typename T>
class PluginRegistrarV3 {
 public:
  PluginRegistrarV3() {
    getPluginRegistry()->registerCreator(instance, "torch_tensorrt");
  }

 private:
  T instance{};
};

#define REGISTER_TORCHTRT_PLUGIN_V3(name) \
  static PluginRegistrarV3<name> pluginRegistrarV3##name {}

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
