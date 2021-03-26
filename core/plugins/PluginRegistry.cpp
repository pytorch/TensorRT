#include "core/util/prelude.h"
#include "core/plugins/impl/interpolate_plugin.h"
#include "core/plugins/impl/normalize_plugin.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"

namespace trtorch{
namespace core{
namespace plugins{
namespace {

using mPluginRegistry = std::unordered_map<std::string, nvinfer1::IPluginCreator*>;

class TRTorchPluginRegistry {
 public:
    bool RegisterPlugins() {
      LOG_DEBUG("==========PLUGINS BEING INITIALIZED=============");
      auto trtorch_logger = util::logging::TRTorchLogger("[TRTorch Plugins Context] - ",
          util::logging::get_logger().get_reportable_severity(),
          util::logging::get_logger().get_is_colored_output_on());

      // Initialize TensorRT built-in plugins
      initLibNvInferPlugins(&trtorch_logger, "");
      int numCreators = 0;
      auto tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);
      for (int k = 0; k < numCreators; ++k)
      {
           if (!tmpList[k])
           {
               std::cout << "Plugin Creator for plugin " << k << " is a nullptr." << std::endl;
               continue;
           }
           std::string pluginName = tmpList[k]->getPluginName();
           plugin_creator_registry[pluginName] = tmpList[k];
           LOG_DEBUG("Register plugin: " << pluginName);
      }


      plugin_creator_registry["InterpolatePlugin"] = new InterpolatePluginCreator();
      plugin_creator_registry["NormalizePlugin"] = new NormalizePluginCreator();
      LOG_DEBUG("Number of plugins: " << plugin_creator_registry.size());
      return true;
    }

    mPluginRegistry get_plugin_creator_registry(){
      return plugin_creator_registry;
    }

 private:
  static mPluginRegistry plugin_creator_registry;
};

TRTorchPluginRegistry& get_trtorch_plugin_registry() {
  static TRTorchPluginRegistry plugin_registry;
  plugin_registry.RegisterPlugins();
  return plugin_registry;
}



} // namespace
}
}
}
