#pragma once
#include <unordered_map>
#include <memory>
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "NvInferRuntime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

// Our own V3 plugin registry that bypasses TensorRT's V2-only registry
class TorchTRTPluginRegistry {
 public:
  static TorchTRTPluginRegistry& getInstance() {
    static TorchTRTPluginRegistry instance;
    return instance;
  }

  template<typename CreatorType>
  void registerCreator(const std::string& plugin_name, const std::string& plugin_version) {
    std::string key = plugin_name + "::" + plugin_version;
    auto creator = std::make_shared<CreatorType>();
    creators_[key] = creator;
    
    // Also register with TensorRT's global registry for deserialization
    auto global_registry = getPluginRegistry();
    if (global_registry) {
      global_registry->registerCreator(*creator, creator->getPluginNamespace());
    }
  }

  nvinfer1::IPluginCreatorV3One* getPluginCreator(
      const std::string& plugin_name, 
      const std::string& plugin_version) {
    std::string key = plugin_name + "::" + plugin_version;
    auto it = creators_.find(key);
    if (it != creators_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  TorchTRTPluginRegistry() = default;
  std::unordered_map<std::string, std::shared_ptr<nvinfer1::IPluginCreatorV3One>> creators_;
};

// Helper class which registers a V3 plugin in our custom registry
template <typename T>
class PluginRegistrar {
 public:
  PluginRegistrar(const std::string& plugin_name, const std::string& plugin_version) {
    TorchTRTPluginRegistry::getInstance().registerCreator<T>(plugin_name, plugin_version);
  }
};

#define REGISTER_TORCHTRT_PLUGIN(CreatorClass, PluginName, PluginVersion) \
  static torch_tensorrt::core::plugins::impl::PluginRegistrar<CreatorClass> \
      pluginRegistrar##CreatorClass {PluginName, PluginVersion}

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
