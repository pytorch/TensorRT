#pragma once
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace plugins {
namespace nvinfer1 {
// Alternate implementation similar to automatic converter registration.
template <typename T>
class PluginRegistrar {
 public:
  PluginRegistrar() {
    getPluginRegistry()->registerCreator(instance, "");
  }

 private:
  T instance{};
};

#define REGISTER_TRTORCH_PLUGIN(name) \
  static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}

} // namespace nvinfer1
} // namespace plugins
} // namespace core
} // namespace trtorch
