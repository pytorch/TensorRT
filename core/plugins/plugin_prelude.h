#pragma once

using mPluginRegistry = std::unordered_map<std::string, nvinfer1::IPluginCreator*>;

class TRTorchPluginRegistry {
 public:
  TRTorchPluginRegistry() = default;
  bool RegisterPlugins();
  mPluginRegistry get_plugin_creator_registry();
};

TRTorchPluginRegistry& get_trtorch_plugin_registry();
