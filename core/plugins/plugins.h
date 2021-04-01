#pragma once
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/util/prelude.h"

namespace trtorch{
namespace core{
namespace plugins{
namespace nvinfer1 {
//Alternate implementation similar to automatic converter registration.
template <typename T>
class PluginRegistrar
{
public:
    PluginRegistrar() { getPluginRegistry()->registerCreator(instance, "trtorch"); }
private:
    T instance{};
};

#define REGISTER_TRTORCH_PLUGIN(name) \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}

} // namespace nvinfer1


// using mPluginRegistry = std::unordered_map<std::string, nvinfer1::IPluginCreator*>;
//
// class TRTorchPluginRegistry {
//  public:
//   TRTorchPluginRegistry() = default;
//   TRTorchPluginRegistry(const TRTorchPluginRegistry&) = delete;
//   TRTorchPluginRegistry& operator=(const TRTorchPluginRegistry&) = delete;
//   TRTorchPluginRegistry(TRTorchPluginRegistry&&) noexcept;
//   TRTorchPluginRegistry& operator=(TRTorchPluginRegistry&&) noexcept;
//   bool RegisterPlugins();
//
//   mPluginRegistry get_plugin_creator_registry();
//
//   mPluginRegistry plugin_creator_registry;
// };
//
// TRTorchPluginRegistry& get_plugin_registry();

// extern std::unordered_map<std::string, nvinfer1::IPluginCreator*> mPluginRegistry;

// std::unordered_map<std::string, nvinfer1::IPluginCreator*> registerPlugins();

// bool registerPlugins() {
//   // std::unordered_map<std::string, nvinfer1::IPluginCreator*> mPluginRegistry;
//   LOG_DEBUG("==========PLUGINS BEING INITIALIZED=============");
//   auto trtorch_logger = util::logging::TRTorchLogger("[TRTorch Plugins Context] - ",
//       util::logging::get_logger().get_reportable_severity(),
//       util::logging::get_logger().get_is_colored_output_on());
//
//   // Initialize TensorRT built-in plugins
//   initLibNvInferPlugins(&trtorch_logger, "");
//   int numCreators = 0;
//   auto tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);
//   for (int k = 0; k < numCreators; ++k)
//   {
//        if (!tmpList[k])
//        {
//            std::cout << "Plugin Creator for plugin " << k << " is a nullptr." << std::endl;
//            continue;
//        }
//        std::string pluginName = tmpList[k]->getPluginName();
//        mPluginRegistry[pluginName] = tmpList[k];
//        LOG_DEBUG("Register plugin: " << pluginName);
//   }
//
//
//   mPluginRegistry["InterpolatePlugin"] = new InterpolatePluginCreator();
//   mPluginRegistry["NormalizePlugin"] = new NormalizePluginCreator();
//   LOG_DEBUG("Number of plugins registered: " << mPluginRegistry.size());
//   return true;
// }

}
}
}
